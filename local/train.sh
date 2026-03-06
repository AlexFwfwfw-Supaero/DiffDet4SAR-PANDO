#!/bin/bash
###############################################################################
# LOCAL — DiffDet4SAR Training (all 50 classes)
#
# Runs on this machine with the torch-env conda environment and NVIDIA RTX 3070.
# Adapted from pando/train.slurm.
#
# Usage:
#   cd /media/alexandre/E6AE9051AE901BDD/PIE\ Code/ATR/ATR-Segmentation/DiffDet4SAR
#   bash local/train.sh
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"    # DiffDet4SAR/
DATA_DIR="$(dirname "$PROJECT_DIR")/ATRNet-STAR-data"

echo "=============================================="
echo " DiffDet4SAR Training — Local (RTX 3070)"
echo "=============================================="
echo " Started: $(date)"
echo " Project: $PROJECT_DIR"
echo " Data:    $DATA_DIR"
echo "=============================================="

# --- Activate conda environment ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env

# --- Environment ---
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export ATRNET_DATA_DIR="$DATA_DIR"
echo "ATRNET_DATA_DIR: $ATRNET_DATA_DIR"
echo ""

# --- GPU info ---
echo ">>> GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not found"
echo ""

cd "$PROJECT_DIR"
mkdir -p output_atrnet_star

echo "Starting training (50 classes, RTX 3070 config)..."
echo ""

# Uses res50 config: batch=2, proposals=300, reduced resolution for 8 GB VRAM.
python train_net.py \
    --config-file configs/diffdet.atrnet.res50.yaml \
    --num-gpus 1 \
    OUTPUT_DIR output_atrnet_star \
    2>&1 | tee output_atrnet_star/training.log

echo ""
echo "=============================================="
echo " Training Complete! $(date)"
echo "=============================================="
