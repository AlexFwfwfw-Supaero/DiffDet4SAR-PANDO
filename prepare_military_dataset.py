#!/usr/bin/env python3
"""
Generate military-only COCO annotation files from SOC_50classes.

Reads the SOC_50classes train.json / test.json, keeps only the 10 military
vehicle categories, and writes:
  <annotation_dir>/train_military.json
  <annotation_dir>/test_military.json

These filtered files share the same image directory as SOC_50classes so no
copying of images is required.

Usage:
  python prepare_military_dataset.py
  python prepare_military_dataset.py --data-dir /path/to/ATRNet-STAR-data
"""

import argparse
import json
import os
from pathlib import Path

MILITARY_NAMES = {
    "2S1", "BMP2", "BRDM_2", "BTR_60", "BTR70",
    "D7", "T62", "T72", "ZIL131", "ZSU_23_4",
}


def filter_split(src_json: Path, dst_json: Path) -> None:
    print(f"  Reading  {src_json}")
    with open(src_json) as f:
        data = json.load(f)

    # Keep only military categories; preserve original IDs.
    mil_cats = [c for c in data["categories"] if c["name"] in MILITARY_NAMES]
    mil_ids  = {c["id"] for c in mil_cats}
    if not mil_cats:
        raise ValueError(f"No military categories found in {src_json}. "
                         "Make sure you are using the SOC_50classes annotation file.")

    # Filter annotations.
    mil_anns = [a for a in data["annotations"] if a["category_id"] in mil_ids]

    # Keep only images that actually appear in the filtered annotations.
    img_ids_with_ann = {a["image_id"] for a in mil_anns}
    mil_imgs = [img for img in data["images"] if img["id"] in img_ids_with_ann]

    out = {
        "info":        data.get("info", {}),
        "licenses":    data.get("licenses", []),
        "categories":  mil_cats,
        "images":      mil_imgs,
        "annotations": mil_anns,
    }

    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_json, "w") as f:
        json.dump(out, f)

    print(f"  Wrote    {dst_json}")
    print(f"           {len(mil_cats)} categories | "
          f"{len(mil_imgs)} images | "
          f"{len(mil_anns)} annotations")
    ann_by_class = {}
    for a in mil_anns:
        cid = a["category_id"]
        name = next(c["name"] for c in mil_cats if c["id"] == cid)
        ann_by_class[name] = ann_by_class.get(name, 0) + 1
    for name in sorted(ann_by_class):
        print(f"    {name:<12}: {ann_by_class[name]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=None,
        help="Path to ATRNet-STAR-data root. "
             "Defaults to $ATRNET_DATA_DIR or ../ATRNet-STAR-data relative to this script.",
    )
    args = parser.parse_args()

    if args.data_dir:
        base = Path(args.data_dir).resolve()
    elif os.environ.get("ATRNET_DATA_DIR"):
        base = Path(os.environ["ATRNET_DATA_DIR"]).resolve()
    else:
        base = (Path(__file__).parent / ".." / "ATRNet-STAR-data").resolve()

    ann_dir = base / "Ground_Range" / "annotation_coco" / "SOC_50classes" / "annotations"

    print("=" * 60)
    print(" Generating military-only annotation files")
    print("=" * 60)
    print(f"Source dir : {ann_dir}")
    print()

    for split in ("train", "test"):
        print(f"[{split}]")
        filter_split(ann_dir / f"{split}.json", ann_dir / f"{split}_military.json")
        print()

    print("Done. Register these with 'atrnet_military_train' / 'atrnet_military_test'")
    print("in register_atrnet.py (already done if using the updated version).")


if __name__ == "__main__":
    main()
