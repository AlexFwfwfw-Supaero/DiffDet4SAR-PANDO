#!/bin/bash
###############################################################################
# PANDO Supercomputer - Military Vehicle Training from Scratch (GPU 2 of 2)
#
# Trains from the ImageNet-pretrained backbone only (no ATRNet checkpoint).
# Use this as the control experiment against the fine-tuned run.
#
# Submit with:  sbatch pando/train_military_scratch.slurm
# Run both jobs at once (one per GPU):
#   sbatch pando/train_military_finetune.slurm
#   sbatch pando/train_military_scratch.slurm
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
echo " DiffDet4SAR — Military Training from Scratch"
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

# --- Generate military-only annotation files (idempotent; safe if already done) ---
echo ""
echo ">>> Preparing military-only annotation files..."
python prepare_military_dataset.py
echo ""

# --- Train from scratch ---
# More iterations than fine-tune because the entire network starts cold.
#   40 000 → first LR decay
#   52 000 → second LR decay
#   60 000 → end
echo ">>> Starting training from ImageNet pretrained weights only..."
echo ""
python train_net.py \
    --config-file configs/diffdet.atrnet.military.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS "detectron2://ImageNetPretrained/torchvision/R-50.pkl" \
    OUTPUT_DIR output_military_scratch \
    SOLVER.MAX_ITER 60000 \
    SOLVER.STEPS "(40000,52000)" \
    2>&1 | tee output_military_scratch/training.log

echo ""
echo "=============================================="
echo " Training Complete! $(date)"
echo "=============================================="
