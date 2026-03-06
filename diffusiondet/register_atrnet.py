"""
ATRNet-STAR Dataset Registration for DiffusionDet

This file registers the ATRNet-STAR dataset with Detectron2.

Three dataset variants are registered:
  atrnet_star_train / atrnet_star_test
      SOC_50classes — all 50 vehicle types (40 civilian + 10 military)

  atrnet_military_train / atrnet_military_test
      Military-only subset of SOC_50classes (10 classes: 2S1, BMP2, BRDM_2,
      BTR_60, BTR70, D7, T62, T72, ZIL131, ZSU_23_4).
      Requires running prepare_military_dataset.py once to generate the
      filtered annotation JSONs (*_military.json).

Supports both local and PANDO supercomputer layouts:
  - Local:  .../ATR-Segmentation/DiffDet4SAR/  +  .../ATR-Segmentation/ATRNet-STAR-data/
  - PANDO:  ~/DiffDet4SAR-project/DiffDet4SAR/ +  ~/DiffDet4SAR-project/ATRNet-STAR-data/

Override with env var ATRNET_DATA_DIR if needed.
"""

import os
from detectron2.data.datasets import register_coco_instances


def _base_path():
    env_data_dir = os.environ.get("ATRNET_DATA_DIR")
    if env_data_dir:
        return os.path.abspath(env_data_dir)
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "ATRNet-STAR-data")
    )


def register_atrnet_star():
    """Register SOC_50classes (all 50 vehicle types)."""
    base_path = _base_path()

    train_image_dir = os.path.join(base_path, "Ground_Range", "Amplitude_8bit", "SOC_50classes", "train")
    test_image_dir  = os.path.join(base_path, "Ground_Range", "Amplitude_8bit", "SOC_50classes", "test")
    annotation_dir  = os.path.join(base_path, "Ground_Range", "annotation_coco", "SOC_50classes", "annotations")
    train_json = os.path.join(annotation_dir, "train.json")
    test_json  = os.path.join(annotation_dir, "test.json")

    register_coco_instances("atrnet_star_train", {}, train_json, train_image_dir)
    register_coco_instances("atrnet_star_test",  {}, test_json,  test_image_dir)

    print(f"✓ Registered atrnet_star (SOC_50classes, 50 classes)")
    print(f"  Train: {train_image_dir}")
    print(f"  Test:  {test_image_dir}")


def register_atrnet_military():
    """
    Register the military-vehicle-only subset (10 classes).

    Annotation files (*_military.json) must exist; generate them with:
        python prepare_military_dataset.py
    Images are shared with the full SOC_50classes split.
    """
    base_path = _base_path()

    image_dir_train = os.path.join(base_path, "Ground_Range", "Amplitude_8bit", "SOC_50classes", "train")
    image_dir_test  = os.path.join(base_path, "Ground_Range", "Amplitude_8bit", "SOC_50classes", "test")
    annotation_dir  = os.path.join(base_path, "Ground_Range", "annotation_coco", "SOC_50classes", "annotations")
    train_json = os.path.join(annotation_dir, "train_military.json")
    test_json  = os.path.join(annotation_dir, "test_military.json")

    if not os.path.exists(train_json) or not os.path.exists(test_json):
        raise FileNotFoundError(
            f"Military annotation files not found:\n  {train_json}\n  {test_json}\n"
            "Run:  python prepare_military_dataset.py"
        )

    register_coco_instances("atrnet_military_train", {}, train_json, image_dir_train)
    register_coco_instances("atrnet_military_test",  {}, test_json,  image_dir_test)

    print(f"✓ Registered atrnet_military (SOC_50classes military subset, 10 classes)")
    print(f"  Train: {image_dir_train}")
    print(f"  Test:  {image_dir_test}")


# Auto-register on import
register_atrnet_star()
register_atrnet_military()
