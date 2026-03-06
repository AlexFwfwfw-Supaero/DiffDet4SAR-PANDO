#!/bin/bash
###############################################################################
# LOCAL — Visualize detections
#
# Mirrors pando/visualize.slurm.
# Picks the latest checkpoint from the given run directory and runs
# visualize_detections.py on a sample of test images.
#
# Usage:
#   cd /media/alexandre/E6AE9051AE901BDD/PIE\ Code/ATR/ATR-Segmentation/DiffDet4SAR
#   bash local/visualize.sh                          # default: output_atrnet_star
#   bash local/visualize.sh output_military_finetune
#   bash local/visualize.sh output_military_scratch
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$(dirname "$PROJECT_DIR")/ATRNet-STAR-data"

OUTPUT_RUN="${1:-output_atrnet_star}"

echo "=============================================="
echo " DiffDet4SAR Visualization — Local (RTX 3070)"
echo "=============================================="
echo " Started:   $(date)"
echo " Run dir:   $OUTPUT_RUN"
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

# --- Find checkpoint ---
CKPT_DIR="$OUTPUT_RUN"
if [ -f "${CKPT_DIR}/model_final.pth" ]; then
    WEIGHTS="${CKPT_DIR}/model_final.pth"
else
    WEIGHTS=$(ls -t "${CKPT_DIR}"/model_*.pth 2>/dev/null | head -1)
fi

if [ -z "$WEIGHTS" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}/"
    exit 1
fi
echo "Using weights: $WEIGHTS"

# --- Pick image directory ---
INPUT_DIR="${DATA_DIR}/Ground_Range/Amplitude_8bit/SOC_50classes/test"
echo "Input dir:   $INPUT_DIR"
echo "Image count: $(find "$INPUT_DIR" -name '*.tif' 2>/dev/null | wc -l) .tif files"
echo ""

VIZ_DIR="${OUTPUT_RUN}/visualizations"

python visualize_detections.py \
    --weights "$WEIGHTS" \
    --input-dir "$INPUT_DIR" \
    --data-dir  "$DATA_DIR" \
    --output-dir "$VIZ_DIR" \
    --num-samples 50 \
    --confidence-threshold 0.3

echo ""
echo "=============================================="
echo " Done! $(date)"
echo "=============================================="
echo ""
echo "Visualizations saved to: $VIZ_DIR/"
