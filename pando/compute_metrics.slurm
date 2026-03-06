#!/bin/bash
###############################################################################
# PANDO Supercomputer - Compute Detection Metrics
#
# Submit after eval-only run has produced coco_eval_instances_results.json:
#   sbatch pando/compute_metrics.slurm
#
# Or chain after eval:
#   sbatch --dependency=afterok:<EVAL_JOB_ID> pando/compute_metrics.slurm
###############################################################################

#SBATCH --job-name=diffdet-metrics
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

echo "=============================================="
echo " DiffDet4SAR Metrics - PANDO"
echo "=============================================="
echo " Job ID: $SLURM_JOB_ID"
echo " Started: $(date)"
echo "=============================================="

export PYTHONPATH="$HOME/DiffDet4SAR-project/DiffDet4SAR-PANDO:$PYTHONPATH"
export ATRNET_DATA_DIR="$HOME/DiffDet4SAR-project/DiffDet4SAR-PANDO/ATRNet-STAR-data"

echo "ATRNET_DATA_DIR: $ATRNET_DATA_DIR"

module load python/2023-3
source activate diffdet4sar

cd ~/DiffDet4SAR-project/DiffDet4SAR-PANDO

# ── Find the predictions file ─────────────────────────────────────────────
# COCOEvaluator writes to: <output_dir>/inference/<dataset_name>/coco_eval_instances_results.json
PRED_DIR="output_atrnet_star_pando/inference/atrnet_star_test"
PRED_FILE="${PRED_DIR}/coco_eval_instances_results.json"

if [ ! -f "$PRED_FILE" ]; then
    echo "ERROR: Predictions file not found: $PRED_FILE"
    echo "Make sure you ran --eval-only first:"
    echo "  python train_net.py --config-file configs/diffdet.atrnet.v100.yaml --eval-only MODEL.WEIGHTS <checkpoint>"
    exit 1
fi

echo "Predictions: $PRED_FILE"
echo ""

# ── Run metrics computation ───────────────────────────────────────────────
OUTPUT_DIR="metrics_output"

python compute_metrics.py \
    --predictions "$PRED_FILE" \
    --output-dir  "$OUTPUT_DIR" \
    --iou-threshold 0.5

echo ""
echo "=============================================="
echo " Done! $(date)"
echo "=============================================="
echo ""
echo "Results saved to: ~/DiffDet4SAR-project/DiffDet4SAR-PANDO/${OUTPUT_DIR}/"
echo ""
echo "Download with:"
echo "  scp -r a.jesus@pando:~/DiffDet4SAR-project/DiffDet4SAR-PANDO/${OUTPUT_DIR}/ ."
