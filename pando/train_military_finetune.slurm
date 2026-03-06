#!/bin/bash
###############################################################################
# PANDO Supercomputer - Military Vehicle Fine-tune (GPU 1 of 2)
#
# Fine-tunes from the LATEST checkpoint produced by the 40-class training.
# The backbone/neck weights are reused; the classification head is re-initialised
# because the class count changes from 40 → 10 (detectron2 skips shape-mismatched
# weights automatically and logs a warning — that is expected).
#
# Submit with:  sbatch pando/train_military_finetune.slurm
# Run both jobs at once (one per GPU):
#   sbatch pando/train_military_finetune.slurm
#   sbatch pando/train_military_scratch.slurm
###############################################################################

#SBATCH --job-name=diffdet-mil-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output_military_finetune/%x_%j.out
#SBATCH --error=output_military_finetune/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.jesus

set -e

echo "=============================================="
echo " DiffDet4SAR — Military Fine-tune (from 40-class checkpoint)"
echo "=============================================="
echo " Job ID:    $SLURM_JOB_ID"
echo " Node:      $SLURM_NODELIST"
echo " Started:   $(date)"
echo "=============================================="

# --- Environment ---
module load python/2023-3
source activate diffdet4sar

echo ">>> GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || \
    echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export https_proxy=http://proxy.isae.fr:3128
export http_proxy=http://proxy.isae.fr:3128

export PYTHONPATH="$HOME/DiffDet4SAR-project/DiffDet4SAR-PANDO:$PYTHONPATH"
export ATRNET_DATA_DIR="$HOME/DiffDet4SAR-project/DiffDet4SAR-PANDO/ATRNet-STAR-data"
echo "ATRNET_DATA_DIR: $ATRNET_DATA_DIR"

cd ~/DiffDet4SAR-project/DiffDet4SAR-PANDO

mkdir -p output_military_finetune

# --- Generate military-only annotation files (idempotent) ---
echo ""
echo ">>> Preparing military-only annotation files..."
python prepare_military_dataset.py
echo ""

# --- Find the latest checkpoint from the 40-class training run ---
CKPT_DIR="output_atrnet_star_pando"
LATEST_CKPT=$(ls -t "${CKPT_DIR}"/model_*.pth 2>/dev/null | head -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}/"
    echo "Make sure the 40-class training has produced at least one checkpoint."
    exit 1
fi

echo ">>> Using checkpoint: $LATEST_CKPT"
echo ""

# --- Fine-tune ---
# SOLVER overrides: fewer iterations since the backbone is already pretrained.
#   STEPS and MAX_ITER decay schedule:
#     20 000 → first LR decay
#     26 000 → second LR decay
#     30 000 → end
python train_net.py \
    --config-file configs/diffdet.atrnet.military.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS "${LATEST_CKPT}" \
    OUTPUT_DIR output_military_finetune \
    SOLVER.MAX_ITER 30000 \
    SOLVER.STEPS "(20000,26000)" \
    2>&1 | tee output_military_finetune/training.log

echo ""
echo "=============================================="
echo " Fine-tune Complete! $(date)"
echo "=============================================="
