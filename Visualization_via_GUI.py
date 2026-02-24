#!/usr/bin/env python3
"""
Interactive DiffDet4SAR Visualization GUI
------------------------------------------
Loads model + images, shows detections with an interactive matplotlib GUI.
- Threshold slider: dynamically adjusts confidence until 3-5 vehicles show
- Prev / Next buttons to navigate between images
- Auto-threshold button that finds the right threshold per image
- Dark-themed GUI, labels placed outside boxes

Usage:
    python demo_visualization.py --weights output_atrnet_star/model_0004999.pth
    python demo_visualization.py --weights output_atrnet_star/model_final.pth --num-samples 50
"""

import argparse
import os
import sys
import json
import random
import numpy as np
import cv2
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, os.path.dirname(__file__))
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultPredictor

from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs
import diffusiondet.register_atrnet


# ── Colour palette ───────────────────────────────────────────────────────────
def class_color(cls_id, n_classes=40):
    h = cls_id / n_classes
    return hsv_to_rgb([h, 0.85, 0.95])   # float RGB [0-1]


# ── Config / model ───────────────────────────────────────────────────────────
def build_predictor(weights, config_file, threshold=0.01):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 4
    cfg.freeze()
    return DefaultPredictor(cfg)


def load_categories(data_dir):
    ann = os.path.join(data_dir,
          "Ground_Range/annotation_coco/SOC_40classes/annotations/test.json")
    with open(ann) as f:
        d = json.load(f)
    return {c["id"]: c["name"] for c in d["categories"]}


# ── Inference ────────────────────────────────────────────────────────────────
def run_inference(predictor, image_path):
    """Return (gray_HxW, boxes_Nx4, scores_N, classes_N)."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)
    rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out  = predictor(rgb)
    inst = out["instances"].to("cpu")
    return (
        gray,
        inst.pred_boxes.tensor.numpy(),
        inst.scores.numpy(),
        inst.pred_classes.numpy(),
    )


def auto_threshold(scores, target_lo=3, target_hi=5):
    """Find threshold that yields target_lo..target_hi detections."""
    if len(scores) == 0:
        return 0.05
    s = np.sort(scores)[::-1]
    idx = min(target_hi - 1, len(s) - 1)
    return float(np.clip(s[idx] * 0.99, 0.01, 0.99))


# ── GUI ──────────────────────────────────────────────────────────────────────
class DetectionGUI:
    def __init__(self, image_paths, predictor, categories):
        self.image_paths = image_paths
        self.predictor   = predictor
        self.categories  = categories
        self.idx         = 0
        self._cache      = {}

        # Figure ──────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(14, 9), facecolor="#1e1e2e")
        self.fig.canvas.manager.set_window_title("DiffDet4SAR – Detection Viewer")

        self.ax_img  = self.fig.add_axes([0.01, 0.18, 0.64, 0.79])
        self.ax_info = self.fig.add_axes([0.67, 0.18, 0.31, 0.79])
        for ax in (self.ax_img, self.ax_info):
            ax.set_facecolor("#1e1e2e")

        # Threshold slider ────────────────────────────────────────────────
        ax_sl = self.fig.add_axes([0.10, 0.08, 0.50, 0.03], facecolor="#313244")
        self.slider = Slider(ax_sl, "Threshold", 0.01, 0.99,
                             valinit=0.30, color="#cba6f7")
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.slider.on_changed(lambda v: self._render())

        # Buttons ─────────────────────────────────────────────────────────
        bs = dict(color="#313244", hovercolor="#45475a")
        self.btn_prev = Button(self.fig.add_axes([0.10, 0.01, 0.09, 0.05]), "◀ Prev", **bs)
        self.btn_next = Button(self.fig.add_axes([0.21, 0.01, 0.09, 0.05]), "Next ▶", **bs)
        self.btn_auto = Button(self.fig.add_axes([0.32, 0.01, 0.12, 0.05]), "Auto 3-5 ✦", **bs)
        self.btn_rand = Button(self.fig.add_axes([0.46, 0.01, 0.09, 0.05]), "Random", **bs)

        for b in (self.btn_prev, self.btn_next, self.btn_auto, self.btn_rand):
            b.label.set_color("white")

        self.btn_prev.on_clicked(lambda e: self._go(self.idx - 1))
        self.btn_next.on_clicked(lambda e: self._go(self.idx + 1))
        self.btn_rand.on_clicked(lambda e: self._go(random.randrange(len(self.image_paths))))
        self.btn_auto.on_clicked(self._do_auto)

        self._render()
        plt.show()

    # ── helpers ──────────────────────────────────────────────────────────
    def _get(self):
        p = self.image_paths[self.idx]
        if p not in self._cache:
            print(f"Running inference on {Path(p).name} …")
            self._cache[p] = run_inference(self.predictor, p)
        return self._cache[p]

    def _go(self, idx):
        self.idx = idx % len(self.image_paths)
        self._render()

    def _do_auto(self, _event):
        _, _, scores, _ = self._get()
        self.slider.set_val(auto_threshold(scores))   # triggers _render

    def _render(self):
        gray, boxes, scores, classes = self._get()
        thresh = self.slider.val
        keep = scores >= thresh
        boxes_f, scores_f, cls_f = boxes[keep], scores[keep], classes[keep]

        # ── image panel ──────────────────────────────────────────────────
        self.ax_img.cla()
        self.ax_img.imshow(gray, cmap="gray", interpolation="nearest",
                           aspect="equal")
        self.ax_img.set_title(
            f"[{self.idx+1}/{len(self.image_paths)}]  {Path(self.image_paths[self.idx]).name}"
            f"   —   {len(boxes_f)} detections   (threshold = {thresh:.2f})",
            color="white", fontsize=9, pad=5)
        self.ax_img.axis("off")

        h, w = gray.shape
        seen_classes = {}

        for box, score, cls_id in zip(boxes_f, scores_f, cls_f):
            x1, y1, x2, y2 = box
            color = class_color(int(cls_id))
            cat   = self.categories.get(int(cls_id) + 1, f"cls_{cls_id}")
            seen_classes[cat] = color

            # Box with semi-transparent fill
            self.ax_img.add_patch(mpatches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.4, edgecolor=color,
                facecolor=(*color, 0.10),
                boxstyle="square,pad=0",
            ))

            # Score label – placed above the box, clamped inside image
            lx = float(np.clip(x1, 1, w - 30))
            ly = float(np.clip(y1 - 2, 6, h - 4))
            self.ax_img.text(
                lx, ly, f"{score:.2f}",
                color="white", fontsize=6, fontweight="bold",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15",
                          fc=color, ec="none", alpha=0.88),
            )

        # ── info panel ───────────────────────────────────────────────────
        self.ax_info.cla()
        self.ax_info.set_facecolor("#1e1e2e")
        self.ax_info.axis("off")

        self.ax_info.text(0.05, 0.97, "Detected classes",
                          color="white", fontsize=11, fontweight="bold",
                          va="top", transform=self.ax_info.transAxes)

        y = 0.91
        for name, color in sorted(seen_classes.items()):
            self.ax_info.add_patch(mpatches.Rectangle(
                (0.05, y - 0.015), 0.08, 0.025,
                transform=self.ax_info.transAxes,
                facecolor=color, edgecolor="none"))
            self.ax_info.text(0.16, y - 0.002, name,
                              color="white", fontsize=8.5,
                              va="center",
                              transform=self.ax_info.transAxes)
            y -= 0.048
            if y < 0.12:
                self.ax_info.text(0.05, y, "…", color="#a6adc8",
                                  fontsize=9, transform=self.ax_info.transAxes)
                break

        if not seen_classes:
            self.ax_info.text(0.05, 0.85, "No detections\nat this threshold.\n\nTry clicking\n\u201cAuto 3-5\u201d.",
                              color="#a6adc8", fontsize=9, va="top",
                              transform=self.ax_info.transAxes)

        if len(scores):
            self.ax_info.text(
                0.05, 0.10,
                f"All predictions : {len(scores)}\n"
                f"Shown (≥{thresh:.2f}) : {len(boxes_f)}\n"
                f"Max score : {scores.max():.3f}",
                color="#a6adc8", fontsize=8, va="bottom",
                transform=self.ax_info.transAxes)

        self.fig.canvas.draw_idle()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config-file",
                        default="configs/diffdet.atrnet.res50.yaml")
    parser.add_argument("--data-dir", default=None,
                        help="ATRNet-STAR-data root (auto-detected if unset)")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()

    # Resolve data dir ────────────────────────────────────────────────────
    data_dir = (args.data_dir
                or os.environ.get("ATRNET_DATA_DIR")
                or os.path.join(os.path.dirname(__file__), "..", "ATRNet-STAR-data"))
    data_dir = os.path.abspath(data_dir)

    # Image list ──────────────────────────────────────────────────────────
    test_img_dir = os.path.join(
        data_dir, "Ground_Range", "Amplitude_8bit", "SOC_40classes", "test")
    image_paths = sorted(Path(test_img_dir).glob("*.tif"))
    if not image_paths:
        sys.exit(f"No .tif files found in {test_img_dir}")
    random.shuffle(image_paths)
    image_paths = [str(p) for p in image_paths[:args.num_samples]]
    print(f"Loaded {len(image_paths)} images")

    categories = load_categories(data_dir)
    print(f"Loaded {len(categories)} categories")

    print(f"Loading model from {args.weights} …")
    predictor = build_predictor(args.weights, args.config_file)
    print("Model ready — opening GUI")

    plt.rcParams.update({
        "figure.facecolor": "#1e1e2e",
        "text.color": "white",
    })
    DetectionGUI(image_paths, predictor, categories)


if __name__ == "__main__":
    main()

