#!/usr/bin/env python3
"""
Compute detection metrics for DiffDet4SAR on ATRNet-STAR.

Usage (after running --eval-only which saves coco_eval_instances_results.json):

  python compute_metrics.py \
      --predictions output_atrnet_star_pando/inference/atrnet_star_test/coco_eval_instances_results.json \
      --annotations $ATRNET_DATA_DIR/Ground_Range/annotation_coco/SOC_40classes/annotations/test.json \
      --output-dir metrics_output

Outputs (saved to --output-dir):
  - per_class_ap.csv       : per-class AP50, AP75, AP50:95
  - summary_metrics.txt    : overall mAP numbers
  - confusion_matrix.png   : normalised confusion matrix
  - per_class_ap.png       : bar chart of per-class AP50
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    sys.exit("pycocotools is required: pip install pycocotools")


# ─────────────────────────────────────────────────────────────────────────────
# AP metrics via pycocotools
# ─────────────────────────────────────────────────────────────────────────────

def compute_coco_metrics(gt_json: str, pred_json: str):
    """Return COCOeval object (already evaluated) + ordered category list."""
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator, coco_gt


def per_class_ap(evaluator: "COCOeval", coco_gt: "COCO"):
    """
    Extract AP50, AP75, AP50:95 per category.
    Returns list of dicts sorted by category id.
    """
    precision = evaluator.eval["precision"]
    # shape: [T, R, K, A, M]
    # T = iou thresholds (10), R = recall pts (101), K = categories, A = area, M = maxDet

    iou_thresholds = evaluator.params.iouThrs  # [0.5, 0.55, ..., 0.95]
    idx_50 = np.where(np.isclose(iou_thresholds, 0.50))[0][0]
    idx_75 = np.where(np.isclose(iou_thresholds, 0.75))[0][0]

    cat_ids = evaluator.params.catIds
    id_to_name = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}

    rows = []
    for k, cat_id in enumerate(cat_ids):
        ap_50_95 = np.mean(precision[:, :, k, 0, -1])
        ap_50    = np.mean(precision[idx_50, :, k, 0, -1])
        ap_75    = np.mean(precision[idx_75, :, k, 0, -1])

        # -1 means not evaluated
        rows.append({
            "cat_id":   cat_id,
            "name":     id_to_name.get(cat_id, str(cat_id)),
            "AP50:95":  float(ap_50_95) if ap_50_95 >= 0 else float("nan"),
            "AP50":     float(ap_50)    if ap_50    >= 0 else float("nan"),
            "AP75":     float(ap_75)    if ap_75    >= 0 else float("nan"),
        })

    rows.sort(key=lambda x: x["cat_id"])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_confusion_matrix(gt_json: str, pred_json: str, iou_threshold: float = 0.5):
    """
    Build an (N+1) x (N+1) confusion matrix where index N is 'background'.
    Rows  = ground-truth class (+ background = missed detections)
    Cols  = predicted class    (+ background = false positives)
    """
    coco_gt = COCO(gt_json)
    with open(pred_json) as f:
        predictions = json.load(f)

    cat_ids = sorted(coco_gt.getCatIds())
    n_cls = len(cat_ids)
    cat_to_idx = {c: i for i, c in enumerate(cat_ids)}
    id_to_name = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}

    cm = np.zeros((n_cls + 1, n_cls + 1), dtype=np.int64)  # +1 for background

    # Group predictions by image
    preds_by_image = {}
    for p in predictions:
        preds_by_image.setdefault(p["image_id"], []).append(p)

    img_ids = coco_gt.getImgIds()
    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)
        preds   = preds_by_image.get(img_id, [])

        if not gt_anns and not preds:
            continue

        # Convert GT boxes xywh → xyxy
        gt_boxes  = np.array([[a["bbox"][0], a["bbox"][1],
                                a["bbox"][0] + a["bbox"][2],
                                a["bbox"][1] + a["bbox"][3]] for a in gt_anns], dtype=float)
        gt_cats   = np.array([a["category_id"] for a in gt_anns])

        # Convert pred boxes xywh → xyxy
        if preds:
            pd_boxes  = np.array([[p["bbox"][0], p["bbox"][1],
                                    p["bbox"][0] + p["bbox"][2],
                                    p["bbox"][1] + p["bbox"][3]] for p in preds], dtype=float)
            pd_cats   = np.array([p["category_id"] for p in preds])
            pd_scores = np.array([p["score"] for p in preds])
        else:
            pd_boxes  = np.zeros((0, 4))
            pd_cats   = np.array([], dtype=int)
            pd_scores = np.array([])

        matched_gt  = set()
        matched_pd  = set()

        if len(gt_boxes) and len(pd_boxes):
            iou_mat = _box_iou(gt_boxes, pd_boxes)  # [G, P]

            # Greedy matching: highest IoU first
            flat_idx = np.argsort(-iou_mat.ravel())
            for fi in flat_idx:
                gi, pi = divmod(fi, len(pd_boxes))
                if iou_mat[gi, pi] < iou_threshold:
                    break
                if gi in matched_gt or pi in matched_pd:
                    continue
                matched_gt.add(gi)
                matched_pd.add(pi)
                gt_idx = cat_to_idx[gt_cats[gi]]
                pd_idx = cat_to_idx[pd_cats[pi]]
                cm[gt_idx, pd_idx] += 1

        # Unmatched GT → background column (missed detection)
        for gi in range(len(gt_anns)):
            if gi not in matched_gt:
                gt_idx = cat_to_idx[gt_cats[gi]]
                cm[gt_idx, n_cls] += 1

        # Unmatched predictions → background row (false positive)
        for pi in range(len(preds)):
            if pi not in matched_pd:
                pd_idx = cat_to_idx[pd_cats[pi]]
                cm[n_cls, pd_idx] += 1

    names = [id_to_name[c] for c in cat_ids] + ["background"]
    return cm, names


def _box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of xyxy boxes. Returns [A, B] matrix."""
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a  = (ax2 - ax1) * (ay2 - ay1)
    area_b  = (bx2 - bx1) * (by2 - by1)
    union   = area_a[:, None] + area_b[None, :] - inter

    return inter / np.where(union > 0, union, 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_ap(rows, output_path):
    names  = [r["name"] for r in rows]
    ap50   = [r["AP50"]    * 100 for r in rows]
    ap5095 = [r["AP50:95"] * 100 for r in rows]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.45), 6))
    ax.bar(x - 0.2, ap50,   0.38, label="AP50",     color="#4c72b0")
    ax.bar(x + 0.2, ap5095, 0.38, label="AP50:95",  color="#dd8452")

    mean_ap50   = np.nanmean(ap50)
    mean_ap5095 = np.nanmean(ap5095)
    ax.axhline(mean_ap50,   color="#4c72b0", ls="--", lw=1.2, alpha=0.7,
               label=f"mAP50 = {mean_ap50:.1f}%")
    ax.axhline(mean_ap5095, color="#dd8452", ls="--", lw=1.2, alpha=0.7,
               label=f"mAP50:95 = {mean_ap5095:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("AP (%)")
    ax.set_title("Per-class Average Precision — ATRNet-STAR (40 classes)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_confusion_matrix(cm, names, output_path, normalise=True):
    # For readability, show only class-vs-class (drop background row/col for the heatmap body,
    # but note background counts as a separate row/column)
    n = len(names)  # includes background

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums.astype(float), 0.0)
        fmt = ".2f"
        title_suffix = " (row-normalised)"
    else:
        cm_plot = cm.astype(float)
        fmt = "d"
        title_suffix = ""

    fig, ax = plt.subplots(figsize=(max(12, n * 0.35), max(10, n * 0.32)))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalise else None)
    plt.colorbar(im, ax=ax, fraction=0.03)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    fontsize = max(4, 9 - n // 10)
    ax.set_xticklabels(names, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(names, fontsize=fontsize)
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix{title_suffix}\nATRNet-STAR — DiffDet4SAR")

    # Annotate cells only if n ≤ 20 (otherwise too crowded)
    if n <= 20:
        thresh = cm_plot.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = cm_plot[i, j]
                text = f"{val:{fmt}}" if val > 0 else ""
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if val > thresh else "black", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute detection metrics for DiffDet4SAR")
    parser.add_argument("--predictions", required=True,
                        help="Path to coco_eval_instances_results.json from COCOEvaluator")
    parser.add_argument("--annotations",
                        default=None,
                        help="Path to COCO annotation JSON (test.json). "
                             "Defaults to $ATRNET_DATA_DIR/Ground_Range/annotation_coco/"
                             "SOC_40classes/annotations/test.json")
    parser.add_argument("--output-dir", default="metrics_output",
                        help="Directory to save plots and CSV")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for confusion matrix matching (default 0.5)")
    args = parser.parse_args()

    # Resolve annotation path
    if args.annotations is None:
        data_dir = os.environ.get("ATRNET_DATA_DIR")
        if not data_dir:
            script_dir = Path(__file__).parent
            data_dir = (script_dir / ".." / "ATRNet-STAR-data").resolve()
        args.annotations = str(
            Path(data_dir) / "Ground_Range" / "annotation_coco"
            / "SOC_40classes" / "annotations" / "test.json"
        )

    print(f"Annotations : {args.annotations}")
    print(f"Predictions : {args.predictions}")

    if not os.path.exists(args.annotations):
        sys.exit(f"Annotation file not found: {args.annotations}")
    if not os.path.exists(args.predictions):
        sys.exit(f"Predictions file not found: {args.predictions}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. COCO mAP metrics ──────────────────────────────────────────────────
    print("\n── COCO AP Metrics ─────────────────────────────────────────────")
    evaluator, coco_gt = compute_coco_metrics(args.annotations, args.predictions)
    rows = per_class_ap(evaluator, coco_gt)

    # Print table
    header = f"{'Class':<25} {'AP50:95':>9} {'AP50':>9} {'AP75':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        ap5095 = f"{r['AP50:95']*100:6.1f}%" if not np.isnan(r['AP50:95']) else "   n/a"
        ap50   = f"{r['AP50']*100:6.1f}%"    if not np.isnan(r['AP50'])    else "   n/a"
        ap75   = f"{r['AP75']*100:6.1f}%"    if not np.isnan(r['AP75'])    else "   n/a"
        print(f"{r['name']:<25} {ap5095:>9} {ap50:>9} {ap75:>9}")

    mean_ap50   = np.nanmean([r["AP50"]    for r in rows]) * 100
    mean_ap5095 = np.nanmean([r["AP50:95"] for r in rows]) * 100
    mean_ap75   = np.nanmean([r["AP75"]    for r in rows]) * 100
    print("-" * len(header))
    print(f"{'mAP':<25} {mean_ap5095:8.2f}% {mean_ap50:8.2f}% {mean_ap75:8.2f}%")

    # Save CSV
    csv_path = out / "per_class_ap.csv"
    with open(csv_path, "w") as f:
        f.write("class,AP50:95,AP50,AP75\n")
        for r in rows:
            f.write(f"{r['name']},{r['AP50:95']:.4f},{r['AP50']:.4f},{r['AP75']:.4f}\n")
    print(f"\nSaved: {csv_path}")

    # Save summary text
    summary_path = out / "summary_metrics.txt"
    with open(summary_path, "w") as f:
        f.write("DiffDet4SAR — ATRNet-STAR evaluation summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"mAP50:95  : {mean_ap5095:.2f}%\n")
        f.write(f"mAP50     : {mean_ap50:.2f}%\n")
        f.write(f"mAP75     : {mean_ap75:.2f}%\n")
        f.write("\nPer-class AP50:\n")
        for r in rows:
            f.write(f"  {r['name']:<25} {r['AP50']*100:6.2f}%\n")
    print(f"Saved: {summary_path}")

    # Plot bar chart
    plot_per_class_ap(rows, out / "per_class_ap.png")

    # ── 2. Confusion matrix ──────────────────────────────────────────────────
    print(f"\n── Confusion Matrix (IoU ≥ {args.iou_threshold}) ─────────────────────────────")
    print("Building confusion matrix (this may take a minute for 29k images)...")
    cm, names = build_confusion_matrix(args.annotations, args.predictions, args.iou_threshold)

    # Save raw counts
    cm_csv = out / "confusion_matrix.csv"
    with open(cm_csv, "w") as f:
        f.write("," + ",".join(names) + "\n")
        for i, name in enumerate(names):
            f.write(name + "," + ",".join(str(v) for v in cm[i]) + "\n")
    print(f"Saved: {cm_csv}")

    plot_confusion_matrix(cm, names, out / "confusion_matrix.png", normalise=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
