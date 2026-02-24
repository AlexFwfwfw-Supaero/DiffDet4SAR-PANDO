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

# --- Environment setup ---
module load python/2023-3
source activate diffdet4sar

# --- Print GPU info (after modules loaded, with fallback) ---
echo ">>> GPU Info:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "  nvidia-smi not found"
    echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "  SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
fi
echo ""

# Set proxy (needed for downloading pretrained weights on first run)
export https_proxy=http://proxy.isae.fr:3128
export http_proxy=http://proxy.isae.fr:3128

# Navigate to project directory
cd ~/DiffDet4SAR-project/DiffDet4SAR-PANDO

# Add DiffDet4SAR/ to Python path so `import detectron2` and `import fvcore` work
# (detectron2/ and fvcore/ are packages directly inside DiffDet4SAR/)
export PYTHONPATH="$HOME/DiffDet4SAR-project/DiffDet4SAR-PANDO:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Tell register_atrnet.py where the dataset lives
export ATRNET_DATA_DIR="$HOME/DiffDet4SAR-project/ATRNet-STAR-data"
echo "ATRNET_DATA_DIR: $ATRNET_DATA_DIR"

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
