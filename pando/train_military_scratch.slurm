#!/bin/bash
###############################################################################
# PANDO Supercomputer - Full 50-Class Training from Scratch
#
# Trains on the complete SOC_50classes dataset (40 civilian + 10 military)
# from the ImageNet-pretrained backbone only (no existing ATRNet checkpoint).
#
# This produces a 50-class checkpoint that can then be used as the ideal
# warm-start for fine-tuning on military-only classes.
#
# Submit with:  sbatch pando/train_military_scratch.slurm
###############################################################################

#SBATCH --job-name=diffdet-50cls-sc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output_atrnet_star_scratch/%x_%j.out
#SBATCH --error=output_atrnet_star_scratch/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.jesus

set -e

echo "=============================================="
echo " DiffDet4SAR — Full 50-Class Training from Scratch"
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

mkdir -p output_atrnet_star_scratch

# --- Train from scratch on all 50 classes ---
# Schedule (V100 32GB, batch=8):
#   60 000 → first LR decay
#   80 000 → second LR decay
#   90 000 → end
# The resulting checkpoint is the ideal warm-start for train_military_finetune.slurm.
echo ">>> Starting 50-class training from ImageNet pretrained weights only..."
echo ""
python train_net.py \
    --config-file configs/diffdet.atrnet.v100.yaml \
    --num-gpus 2 \
    MODEL.WEIGHTS "detectron2://ImageNetPretrained/torchvision/R-50.pkl" \
    OUTPUT_DIR output_atrnet_star_scratch \
    2>&1 | tee output_atrnet_star_scratch/training.log

echo ""
echo "=============================================="
echo " Training Complete! $(date)"
echo " Checkpoint in: output_atrnet_star_scratch/"
echo " Next step: update CKPT_DIR in train_military_finetune.slurm"
echo "   to point to output_atrnet_star_scratch/ for optimal fine-tuning."
echo "=============================================="
