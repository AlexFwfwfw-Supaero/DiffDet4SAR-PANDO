# Environment Setup Guide

## Quick Setup

### Option 1: Conda Environment (Recommended)

```bash
# Create environment
conda create -n diffdet python=3.10 -y
conda activate diffdet

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

### Option 2: Exact Versions (Guaranteed to work)

```bash
conda create -n diffdet python=3.10 -y
conda activate diffdet
pip install -r requirements-exact.txt --index-url https://download.pytorch.org/whl/cu124
```

### Option 3: PANDO Supercomputer

The setup is automated - just run:
```bash
bash pando/setup_environment.sh
```

---

## Package Details

### Essential (Required for Training)
- **torch, torchvision, torchaudio** - PyTorch deep learning framework
- **numpy** - Numerical operations
- **pillow** - Image loading
- **opencv-python** - Image processing
- **timm** - Model architectures (used by DiffusionDet)
- **tqdm** - Progress bars
- **pycocotools** - COCO dataset format (ATRNet-STAR uses this)
- **tensorboard** - Training monitoring

### Optional
- **matplotlib** - Only needed for `visualize_detections.py --show` (interactive display)

### NOT Needed (Already Included)
- **detectron2** - Included in-tree at `DiffDet4SAR/detectron2/`
- **fvcore** - Included in-tree at `DiffDet4SAR/fvcore/`
- **scipy** - Not required for DiffDet4SAR

---

## Verification

After installation, verify everything works:

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
import sys
sys.path.insert(0, '.')
from detectron2 import model_zoo
print('Detectron2: OK')
from diffusiondet import DiffusionDet
print('DiffusionDet: OK')
print('✓ All dependencies working!')
"
```

---

## CUDA Version Notes

Current requirements use **CUDA 12.4** (`cu124`). 

For different CUDA versions:
- **CUDA 11.8**: `--index-url https://download.pytorch.org/whl/cu118`
- **CUDA 12.1**: `--index-url https://download.pytorch.org/whl/cu121`
- **CPU only**: `--index-url https://download.pytorch.org/whl/cpu`

Check your CUDA version:
```bash
nvidia-smi  # Look for "CUDA Version"
```

---

## Troubleshooting

**Import error: detectron2 not found**
- Don't pip install detectron2 - it's in-tree
- Make sure you're running from the `DiffDet4SAR/` directory
- The code adds detectron2 to path automatically

**CUDA out of memory**
- Use the configs with smaller batch sizes
- `configs/diffdet.atrnet.res50.yaml` (batch=2 for 8GB GPU)
- Close other GPU applications

**Module 'timm' not found**
```bash
pip install timm
```
