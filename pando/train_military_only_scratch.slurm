#!/bin/bash
###############################################################################
# PANDO Supercomputer - Military-Only Training from Scratch
#
# Trains directly on the 10 military vehicle classes from ImageNet-pretrained
# backbone only (no 50-class or ATRNet checkpoint).
#
# Use this to get a military-class model without the two-stage
# (scratch → finetune) pipeline. Convergence will be slower than fine-tuning
# from a 50-class checkpoint, but no prior run is required.
#
# Run order (standalone):
#   1. sbatch pando/train_military_only_scratch.slurm
#
# Submit with:  sbatch pando/train_military_only_scratch.slurm
###############################################################################

#SBATCH --job-name=diffdet-mil-sc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output_military_scratch/%x_%j.out
#SBATCH --error=output_military_scratch/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.jesus

set -e

echo "=============================================="
echo " DiffDet4SAR — Military-Only Training from Scratch"
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

mkdir -p output_military_scratch

# --- Generate military-only annotation files (idempotent) ---
echo ""
echo ">>> Preparing military-only annotation files..."
python prepare_military_dataset.py
echo ""

# --- Train from scratch on military-only (10 classes) ---
# Schedule (V100 32GB, batch=8, ~3 671 train images):
#   60 000 iters ≈ 131 epochs — more iterations than fine-tune to compensate
#                                for the cold ImageNet-only initialisation.
#   40 000 → first LR decay
#   52 000 → second LR decay
#   60 000 → end
echo ">>> Starting 10-class military training from ImageNet pretrained weights..."
echo ""
python train_net.py \
    --config-file configs/diffdet.atrnet.military.yaml \
    --num-gpus 2 \
    MODEL.WEIGHTS "detectron2://ImageNetPretrained/torchvision/R-50.pkl" \
    OUTPUT_DIR output_military_scratch \
    SOLVER.MAX_ITER 60000 \
    SOLVER.STEPS "(40000,52000)" \
    2>&1 | tee output_military_scratch/training.log

echo ""
echo "=============================================="
echo " Training Complete! $(date)"
echo " Checkpoint in: output_military_scratch/"
echo "=============================================="
