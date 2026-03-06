#!/bin/bash
###############################################################################
# LOCAL — Eval-only: run inference and save predictions JSON
#
# Runs --eval-only on the test set to produce:
#   <output_dir>/inference/atrnet_star_test/coco_eval_instances_results.json
# which is then consumed by compute_metrics.sh.
#
# Usage:
#   cd /media/alexandre/E6AE9051AE901BDD/PIE\ Code/ATR/ATR-Segmentation/DiffDet4SAR
#   bash local/eval.sh                        # uses latest checkpoint in output_atrnet_star/
#   bash local/eval.sh output_military_finetune  # evaluate a different run
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$(dirname "$PROJECT_DIR")/ATRNet-STAR-data"

# Optional first argument: checkpoint directory (default: output_atrnet_star)
OUTPUT_RUN="${1:-output_atrnet_star}"

echo "=============================================="
echo " DiffDet4SAR Eval-only — Local (RTX 3070)"
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
echo ">>> Using weights: $WEIGHTS"
echo ""

# --- Determine which config to use based on output dir ---
if [[ "$OUTPUT_RUN" == *military* ]]; then
    CONFIG="configs/diffdet.atrnet.military.yaml"
else
    CONFIG="configs/diffdet.atrnet.res50.yaml"
fi
echo ">>> Config: $CONFIG"
echo ""

python train_net.py \
    --config-file "$CONFIG" \
    --num-gpus 1 \
    --eval-only \
    MODEL.WEIGHTS "$WEIGHTS" \
    OUTPUT_DIR "$OUTPUT_RUN" \
    INPUT.MIN_SIZE_TEST 800 \
    2>&1 | tee "${OUTPUT_RUN}/eval.log"

echo ""
echo "=============================================="
echo " Eval Complete! $(date)"
echo "=============================================="
echo ""
echo "Predictions saved to:"
echo "  ${OUTPUT_RUN}/inference/*/coco_eval_instances_results.json"
echo ""
echo "Now run:  bash local/compute_metrics.sh $OUTPUT_RUN"
