#!/usr/bin/env python3
"""Verify that SOC_50classes dataset and military annotations are correctly set up."""

import json
import os
import sys
from pathlib import Path

ROOT = Path("/media/alexandre/E6AE9051AE901BDD/PIE Code/ATR/ATR-Segmentation")
DATA_ROOT = ROOT / "ATRNet-STAR-data/Ground_Range"
ANN_DIR = DATA_ROOT / "annotation_coco/SOC_50classes/annotations"
IMG_DIR = DATA_ROOT / "Amplitude_8bit/SOC_50classes"

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def ok(msg):    print(f"  {GREEN}[OK]{RESET}      {msg}")
def fail(msg):  print(f"  {RED}[MISSING]{RESET} {msg}")
def info(msg):  print(f"            {msg}")

all_good = True

print("=" * 60)
print("  SOC_50classes Dataset Verification")
print("=" * 60)

# ── Image directories ─────────────────────────────────────────
print("\n>>> Image directories")
for split in ("train", "test"):
    p = IMG_DIR / split
    if p.exists():
        count = len(list(p.glob("*.tif")))
        ok(f"{split}: {count} .tif files")
    else:
        fail(f"{split}: DIR MISSING")
        all_good = False

# ── Annotation files ──────────────────────────────────────────
print("\n>>> Annotation files")
MILITARY_NAMES_EXPECTED = {
    "2S1", "BMP2", "BRDM_2", "BTR_60", "BTR70",
    "D7",  "T62",  "T72",    "ZIL131", "ZSU_23_4"
}

for fname in ("train.json", "test.json", "train_military.json", "test_military.json"):
    fpath = ANN_DIR / fname
    if fpath.exists():
        with open(fpath) as f:
            d = json.load(f)
        n_cat  = len(d["categories"])
        n_img  = len(d["images"])
        n_ann  = len(d["annotations"])
        ok(fname)
        info(f"categories={n_cat}  images={n_img}  annotations={n_ann}")

        if "military" in fname:
            cat_names = {c["name"] for c in d["categories"]}
            missing_classes = MILITARY_NAMES_EXPECTED - cat_names
            extra_classes   = cat_names - MILITARY_NAMES_EXPECTED
            if missing_classes:
                info(f"{RED}Missing classes: {sorted(missing_classes)}{RESET}")
                all_good = False
            else:
                info(f"All 10 military classes present: {sorted(cat_names)}")
            if extra_classes:
                info(f"{YELLOW}Extra classes: {sorted(extra_classes)}{RESET}")

            # Cross-check: all annotation image_ids exist in images
            img_id_set = {img["id"] for img in d["images"]}
            bad = [a for a in d["annotations"] if a["image_id"] not in img_id_set]
            if bad:
                info(f"{RED}Dangling annotations: {len(bad)}{RESET}")
                all_good = False
            else:
                info("No dangling annotations")

            # Check category IDs are contiguous from 1..10
            ids = sorted(c["id"] for c in d["categories"])
            if ids == list(range(1, 11)):
                info(f"Category IDs: {ids} (1..10 — correct for military-only config)")
            else:
                info(f"{YELLOW}Category IDs: {ids} (not 1..10 — may need re-mapping){RESET}")
    else:
        if "military" in fname:
            fail(f"{fname}  ← run prepare_military_dataset.py first!")
            all_good = False
        else:
            fail(f"{fname}")
            all_good = False

# ── Registration path simulation ─────────────────────────────
print("\n>>> Registration paths (as register_atrnet.py resolves them)")
BASE = DATA_ROOT.parent  # ATRNet-STAR-data
for kind, split_name, img_rel, ann_rel in [
    ("full-50",  "train", "Ground_Range/Amplitude_8bit/SOC_50classes/train",
                          "Ground_Range/annotation_coco/SOC_50classes/annotations/train.json"),
    ("full-50",  "test",  "Ground_Range/Amplitude_8bit/SOC_50classes/test",
                          "Ground_Range/annotation_coco/SOC_50classes/annotations/test.json"),
    ("military", "train", "Ground_Range/Amplitude_8bit/SOC_50classes/train",
                          "Ground_Range/annotation_coco/SOC_50classes/annotations/train_military.json"),
    ("military", "test",  "Ground_Range/Amplitude_8bit/SOC_50classes/test",
                          "Ground_Range/annotation_coco/SOC_50classes/annotations/test_military.json"),
]:
    img_path = BASE / img_rel
    ann_path = BASE / ann_rel
    img_ok = "OK" if img_path.exists() else f"{RED}MISSING{RESET}"
    ann_ok = "OK" if ann_path.exists() else f"{RED}MISSING{RESET}"
    label = f"{kind:8s} {split_name}"
    print(f"  [{label}]  img={img_ok}  ann={ann_ok}")

# ── Config cross-check ────────────────────────────────────────
print("\n>>> Config files")
cfg_dir = ROOT / "DiffDet4SAR/configs"
configs = {
    "diffdet.atrnet.military.yaml": ("NUM_CLASSES: 10", "atrnet_military_train", "atrnet_military_test"),
    "diffdet.atrnet.res50.yaml":    ("NUM_CLASSES: 50",),
    "diffdet.atrnet.v100.yaml":     ("NUM_CLASSES: 50",),
}
for cfg_name, *checks in [(k, v) for k, v in configs.items()]:
    cfg_path = cfg_dir / cfg_name
    if not cfg_path.exists():
        fail(cfg_name)
        all_good = False
        continue
    text = cfg_path.read_text()
    issues = [c for c in configs[cfg_name] if c not in text]
    if issues:
        ok(cfg_name)
        for c in issues:
            info(f"{RED}Expected string not found: '{c}'{RESET}")
        all_good = False
    else:
        ok(f"{cfg_name}  (all checks passed)")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_good:
    print(f"  {GREEN}RESULT: Everything looks good!{RESET}")
else:
    print(f"  {RED}RESULT: Some issues found above — fix before training.{RESET}")
print("=" * 60)
