#!/bin/bash
###############################################################################
# LOCAL — Compute detection metrics
#
# Mirrors pando/compute_metrics.slurm.
# Searches for predictions JSON inside the given run directory, then calls
# compute_metrics.py to produce AP tables, CSV, and confusion-matrix plots.
#
# Usage:
#   cd /media/alexandre/E6AE9051AE901BDD/PIE\ Code/ATR/ATR-Segmentation/DiffDet4SAR
#   bash local/compute_metrics.sh                          # default: output_atrnet_star
#   bash local/compute_metrics.sh output_military_finetune
#   bash local/compute_metrics.sh output_military_scratch
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$(dirname "$PROJECT_DIR")/ATRNet-STAR-data"

OUTPUT_RUN="${1:-output_atrnet_star}"

echo "=============================================="
echo " DiffDet4SAR Metrics — Local"
echo "=============================================="
echo " Started:   $(date)"
echo " Run dir:   $OUTPUT_RUN"
echo "=============================================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env

export ATRNET_DATA_DIR="$DATA_DIR"

cd "$PROJECT_DIR"

# --- Find predictions JSON ---
# COCOEvaluator saves to: <output_dir>/inference/<dataset_name>/coco_eval_instances_results.json
PRED_FILE=$(find "${OUTPUT_RUN}/inference" -name "coco_eval_instances_results.json" 2>/dev/null | head -1)

if [ -z "$PRED_FILE" ]; then
    echo "ERROR: Predictions file not found under ${OUTPUT_RUN}/inference/"
    echo "Run eval first:  bash local/eval.sh $OUTPUT_RUN"
    exit 1
fi
echo "Predictions: $PRED_FILE"

# --- Pick annotation file: military subset or full 50-class ---
if [[ "$OUTPUT_RUN" == *military* ]]; then
    ANN_FILE="${DATA_DIR}/Ground_Range/annotation_coco/SOC_50classes/annotations/test_military.json"
    METRICS_DIR="${OUTPUT_RUN}/metrics_output"
else
    ANN_FILE="${DATA_DIR}/Ground_Range/annotation_coco/SOC_50classes/annotations/test.json"
    METRICS_DIR="${OUTPUT_RUN}/metrics_output"
fi

if [ ! -f "$ANN_FILE" ]; then
    echo "ERROR: Annotation file not found: $ANN_FILE"
    if [[ "$OUTPUT_RUN" == *military* ]]; then
        echo "Run:  python prepare_military_dataset.py"
    fi
    exit 1
fi
echo "Annotations: $ANN_FILE"
echo ""

python compute_metrics.py \
    --predictions "$PRED_FILE" \
    --annotations "$ANN_FILE" \
    --output-dir  "$METRICS_DIR" \
    --iou-threshold 0.5

echo ""
echo "=============================================="
echo " Done! $(date)"
echo "=============================================="
echo ""
echo "Results saved to: $METRICS_DIR/"
