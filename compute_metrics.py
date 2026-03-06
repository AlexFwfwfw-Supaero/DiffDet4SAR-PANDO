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
    import types

    coco_gt = COCO(gt_json)

    # Sanitize predictions: ensure plain Python floats in bbox/score.
    with open(pred_json) as f:
        _preds = json.load(f)
    for p in _preds:
        p["bbox"]  = [float(x) for x in p["bbox"]]
        p["score"] = float(p["score"])

    coco_dt = coco_gt.loadRes(_preds)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")

    # Patch computeIoU with a pure-NumPy implementation that bypasses the
    # Cython maskUtils.iou call.  pycocotools compiled against NumPy 1.x
    # raises "TypeError: Cannot convert numpy.ndarray to numpy.ndarray" when
    # run with NumPy 2.x because of a Cython ABI mismatch.
    def _computeIoU_numpy(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        # Return zero-IoU matrix when one side is empty (avoids reshape crash).
        if len(dt) == 0 or len(gt) == 0:
            return np.zeros((len(dt), len(gt)), dtype=np.float64)

        def _xywh_to_xyxy(boxes):
            """Convert list of [x,y,w,h] to [D,4] xyxy float64 array."""
            return np.array(
                [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes],
                dtype=np.float64,
            )

        d_boxes = _xywh_to_xyxy([d["bbox"] for d in dt])   # [D, 4]
        g_boxes = _xywh_to_xyxy([g["bbox"] for g in gt])   # [G, 4]

        # Compute IoU matrix [D, G]
        inter_x1 = np.maximum(d_boxes[:, 0:1], g_boxes[:, 0])
        inter_y1 = np.maximum(d_boxes[:, 1:2], g_boxes[:, 1])
        inter_x2 = np.minimum(d_boxes[:, 2:3], g_boxes[:, 2])
        inter_y2 = np.minimum(d_boxes[:, 3:4], g_boxes[:, 3])
        inter_w  = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h  = np.maximum(0.0, inter_y2 - inter_y1)
        inter    = inter_w * inter_h                         # [D, G]

        area_d = (d_boxes[:, 2] - d_boxes[:, 0]) * (d_boxes[:, 3] - d_boxes[:, 1])  # [D]
        area_g = (g_boxes[:, 2] - g_boxes[:, 0]) * (g_boxes[:, 3] - g_boxes[:, 1])  # [G]

        # For iscrowd=1 GTs, denominator is detection area (not union)
        iscrowd = np.array([int(o["iscrowd"]) for o in gt], dtype=bool)  # [G]
        union = area_d[:, None] + area_g[None, :] - inter               # [D, G]
        if iscrowd.any():
            union[:, iscrowd] = area_d[:, None]  # broadcast [D,1] → [D, #crowd]

        iou = inter / np.where(union > 0, union, 1e-9)
        return iou

    evaluator.computeIoU = types.MethodType(_computeIoU_numpy, evaluator)

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

# ATRNet-STAR military vehicle classes (IDs vary by annotation file;
# these are resolved dynamically by name matching at runtime).
MILITARY_VEHICLE_NAMES = [
    "2S1", "BMP2", "BRDM_2", "BTR_60", "BTR70",
    "D7", "T62", "T72", "ZIL131", "ZSU_23_4",
]


def build_confusion_matrix(
    gt_json: str,
    pred_json: str,
    iou_threshold: float = 0.5,
    filter_cat_ids=None,
    score_threshold: float = 0.0,
):
    """
    Build an (N+1) x (N+1) confusion matrix where index N is 'background'.
    Rows  = ground-truth class (+ background = missed detections)
    Cols  = predicted class    (+ background = false positives)

    filter_cat_ids : if given, only consider GT/predictions for those category
                     IDs.  The resulting CM covers only those classes.
    score_threshold: discard predictions with score < this value.
    """
    coco_gt = COCO(gt_json)
    with open(pred_json) as f:
        predictions = json.load(f)

    # Optionally restrict to a subset of classes.
    if filter_cat_ids is not None:
        active_cat_ids = sorted(filter_cat_ids)
    else:
        active_cat_ids = sorted(coco_gt.getCatIds())

    active_cat_set = set(active_cat_ids)
    n_cls = len(active_cat_ids)
    cat_to_idx = {c: i for i, c in enumerate(active_cat_ids)}
    id_to_name  = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}

    cm = np.zeros((n_cls + 1, n_cls + 1), dtype=np.int64)  # +1 for background

    # Group predictions by image (apply score threshold here).
    # NOTE: pass imgIds as a list to avoid a pycocotools bug where string image
    # IDs are iterated character-by-character when passed as a bare scalar.
    preds_by_image = {}
    for p in predictions:
        if p["score"] < score_threshold:
            continue
        if filter_cat_ids is not None and p["category_id"] not in active_cat_set:
            continue
        preds_by_image.setdefault(p["image_id"], []).append(p)

    img_ids = coco_gt.getImgIds()
    for img_id in img_ids:
        # ↓ wrap in list — bare string IDs are mishandled by _isArrayLike in
        #   some pycocotools builds, causing getAnnIds to return [].
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gt_anns_all = coco_gt.loadAnns(ann_ids)

        # Filter GT to active classes only.
        gt_anns = [a for a in gt_anns_all if a["category_id"] in active_cat_set]
        preds   = preds_by_image.get(img_id, [])

        if not gt_anns and not preds:
            continue

        # Build GT arrays (xywh → xyxy).
        if gt_anns:
            gt_boxes = np.array(
                [[a["bbox"][0], a["bbox"][1],
                  a["bbox"][0] + a["bbox"][2],
                  a["bbox"][1] + a["bbox"][3]] for a in gt_anns], dtype=float)
            gt_cats  = np.array([a["category_id"] for a in gt_anns])
        else:
            gt_boxes = np.zeros((0, 4))
            gt_cats  = np.array([], dtype=int)

        # Build prediction arrays (xywh → xyxy).
        if preds:
            pd_boxes = np.array(
                [[p["bbox"][0], p["bbox"][1],
                  p["bbox"][0] + p["bbox"][2],
                  p["bbox"][1] + p["bbox"][3]] for p in preds], dtype=float)
            pd_cats  = np.array([p["category_id"] for p in preds])
        else:
            pd_boxes = np.zeros((0, 4))
            pd_cats  = np.array([], dtype=int)

        matched_gt = set()
        matched_pd = set()

        if len(gt_boxes) and len(pd_boxes):
            iou_mat = _box_iou(gt_boxes, pd_boxes)  # [G, P]

            # Greedy matching: highest IoU first.
            flat_idx = np.argsort(-iou_mat.ravel())
            for fi in flat_idx:
                gi, pi = divmod(fi, len(pd_boxes))
                if iou_mat[gi, pi] < iou_threshold:
                    break
                if gi in matched_gt or pi in matched_pd:
                    continue
                matched_gt.add(gi)
                matched_pd.add(pi)
                gt_idx = cat_to_idx[int(gt_cats[gi])]
                pd_idx = cat_to_idx[int(pd_cats[pi])]
                cm[gt_idx, pd_idx] += 1

        # Unmatched GT → background column (missed detection).
        for gi in range(len(gt_anns)):
            if gi not in matched_gt:
                gt_idx = cat_to_idx[int(gt_cats[gi])]
                cm[gt_idx, n_cls] += 1

        # Unmatched predictions → background row (false positive).
        for pi in range(len(preds)):
            if pi not in matched_pd:
                pd_idx = cat_to_idx[int(pd_cats[pi])]
                cm[n_cls, pd_idx] += 1

    names = [id_to_name[c] for c in active_cat_ids] + ["background"]
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


def plot_confusion_matrix(cm, names, output_path, normalise=True, title_extra=""):
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
    ax.set_title(f"Confusion Matrix{title_suffix}{title_extra}\nATRNet-STAR — DiffDet4SAR")

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
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Minimum prediction score for confusion matrix (default 0.0)")
    parser.add_argument("--tank-classes", nargs="+", default=None,
                        metavar="NAME",
                        help="Class names to include in the military-vehicle confusion "
                             "matrix. Defaults to the 10 known military classes in "
                             "ATRNet-STAR (2S1 BMP2 BRDM_2 BTR_60 BTR70 D7 T62 T72 "
                             "ZIL131 ZSU_23_4). Pass a subset to restrict further.")
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

    # ── 2. Confusion matrix — all classes ───────────────────────────────────
    print(f"\n── Confusion Matrix — all classes (IoU ≥ {args.iou_threshold}) ──────────")
    print("Building confusion matrix (this may take a minute for 29k images)...")
    cm_all, names_all = build_confusion_matrix(
        args.annotations, args.predictions,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )

    cm_csv = out / "confusion_matrix.csv"
    with open(cm_csv, "w") as f:
        f.write("," + ",".join(names_all) + "\n")
        for i, name in enumerate(names_all):
            f.write(name + "," + ",".join(str(v) for v in cm_all[i]) + "\n")
    print(f"Saved: {cm_csv}")
    plot_confusion_matrix(cm_all, names_all, out / "confusion_matrix.png", normalise=True)

    # ── 3. Confusion matrix — military vehicles only ─────────────────────────
    # Resolve which category IDs count as military vehicles.
    all_cats = {c["name"]: c["id"] for c in coco_gt.dataset["categories"]}
    requested = args.tank_classes if args.tank_classes is not None else MILITARY_VEHICLE_NAMES
    tank_cat_ids = [all_cats[n] for n in requested if n in all_cats]
    missing = [n for n in requested if n not in all_cats]
    if missing:
        print(f"Warning: tank class names not found in annotation file and skipped: {missing}")

    if tank_cat_ids:
        print(f"\n── Confusion Matrix — military vehicles ({len(tank_cat_ids)} classes, "
              f"IoU ≥ {args.iou_threshold}) ──")
        print("Classes:", [n for n in requested if n in all_cats])
        print("Building military-vehicle confusion matrix...")
        cm_mil, names_mil = build_confusion_matrix(
            args.annotations, args.predictions,
            iou_threshold=args.iou_threshold,
            filter_cat_ids=tank_cat_ids,
            score_threshold=args.score_threshold,
        )

        mil_csv = out / "confusion_matrix_military.csv"
        with open(mil_csv, "w") as f:
            f.write("," + ",".join(names_mil) + "\n")
            for i, name in enumerate(names_mil):
                f.write(name + "," + ",".join(str(v) for v in cm_mil[i]) + "\n")
        print(f"Saved: {mil_csv}")
        plot_confusion_matrix(
            cm_mil, names_mil,
            out / "confusion_matrix_military.png",
            normalise=True,
            title_extra=" — Military Vehicles",
        )
    else:
        print("Warning: no valid tank/military-vehicle classes found; skipping military CM.")

    print("\nDone.")


if __name__ == "__main__":
    main()
