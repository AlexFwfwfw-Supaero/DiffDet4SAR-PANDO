#!/bin/bash
###############################################################################
# PANDO Supercomputer - DiffDet4SAR Training Job (Slurm)
#
# Submit with:  sbatch pando/train.slurm
# Monitor with: squeue -u $USER
# Cancel with:  scancel <JOB_ID>
# View logs:    tail -f diffdet4sar_<JOB_ID>.out
###############################################################################

#SBATCH --job-name=diffdet4sar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -n 12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.jesus@isae-supaero.fr

set -e

echo "=============================================="
echo " DiffDet4SAR Training - PANDO"
echo "=============================================="
echo " Job ID:    $SLURM_JOB_ID"
echo " Node:      $SLURM_NODELIST"
echo " Tasks:     $SLURM_NTASKS"
echo " Started:   $(date)"
echo "=============================================="
echo ""
echo " Useful commands (run from a frontend):"
echo "   Monitor:  squeue -u $USER"
echo "   Cancel:   scancel $SLURM_JOB_ID"
echo "   Logs:     tail -f diffdet4sar_${SLURM_JOB_ID}.out"
echo ""

# --- Print GPU info via nvidia-smi (works at shell level, no Python needed) ---
echo ">>> GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# --- Environment setup ---
module load python/2023-3
source activate diffdet4sar

# Set proxy (needed for downloading pretrained weights on first run)
export https_proxy=http://proxy.isae.fr:3128
export http_proxy=http://proxy.isae.fr:3128

# Navigate to project directory
cd ~/DiffDet4SAR-project/DiffDet4SAR

# Add in-tree detectron2 and fvcore to Python path (not pip-installed on PANDO)
export PYTHONPATH="$HOME/DiffDet4SAR-project/DiffDet4SAR/detectron2:$HOME/DiffDet4SAR-project/DiffDet4SAR/fvcore:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Create output directory
mkdir -p output_atrnet_star_pando

echo "Starting training..."
echo ""

# --- Launch training ---
# Uses the V100-optimized config (batch=8, proposals=500, full resolution)
python train_net.py \
    --config-file configs/diffdet.atrnet.v100.yaml \
    --num-gpus 1 \
    2>&1 | tee output_atrnet_star_pando/training.log

echo ""
echo "=============================================="
echo " Training Complete! $(date)"
echo "=============================================="
