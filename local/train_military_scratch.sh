#!/bin/bash
###############################################################################
# LOCAL — Military Vehicle Training from Scratch
#
# Mirrors pando/train_military_scratch.slurm.
# Runs on this machine with torch-env + RTX 3070 (8 GB VRAM).
#
# Usage:
#   cd /media/alexandre/E6AE9051AE901BDD/PIE\ Code/ATR/ATR-Segmentation/DiffDet4SAR
#   bash local/train_military_scratch.sh
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$(dirname "$PROJECT_DIR")/ATRNet-STAR-data"

echo "=============================================="
echo " DiffDet4SAR — Military Training from Scratch (local, RTX 3070)"
echo "=============================================="
echo " Started: $(date)"
echo " Project: $PROJECT_DIR"
echo " Data:    $DATA_DIR"
echo "=============================================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export ATRNET_DATA_DIR="$DATA_DIR"
echo "ATRNET_DATA_DIR: $ATRNET_DATA_DIR"
echo ""

echo ">>> GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not found"
echo ""

cd "$PROJECT_DIR"
mkdir -p output_military_scratch

# --- Generate military-only annotation files (idempotent) ---
echo ">>> Preparing military-only annotation files..."
python prepare_military_dataset.py
echo ""

echo ">>> Starting training from ImageNet pretrained weights only..."
echo ""

# RTX 3070 (8 GB): batch=2, proposals=300, reduced resolution.
python train_net.py \
    --config-file configs/diffdet.atrnet.military.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS "detectron2://ImageNetPretrained/torchvision/R-50.pkl" \
    OUTPUT_DIR output_military_scratch \
    SOLVER.IMS_PER_BATCH 2 \
    MODEL.DiffusionDet.NUM_PROPOSALS 300 \
    INPUT.MIN_SIZE_TRAIN "(640,800)" \
    INPUT.MIN_SIZE_TEST 800 \
    SOLVER.BASE_LR 0.000005 \
    SOLVER.MAX_ITER 60000 \
    SOLVER.STEPS "(40000,52000)" \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    2>&1 | tee output_military_scratch/training.log

echo ""
echo "=============================================="
echo " Training Complete! $(date)"
echo "=============================================="
