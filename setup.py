"""
Setup script for LCFusion_LT3D project.
Requires: miniconda3 with Python 3.10 (conda env: lcfusion)

Environment: /home/batashey/miniconda3/envs/lcfusion/
Activate:    source /home/batashey/miniconda3/bin/activate lcfusion

Usage (from scratch):
  bash /home/batashey/miniconda3/bin/activate lcfusion
  python setup.py
"""

import subprocess, sys, os

def run(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (result.stdout + result.stderr)[-2000:]
    if out.strip():
        print(out)
    return result.returncode

PYTHON = sys.executable
PIP = f"{PYTHON} -m pip"

# ── PyTorch nightly + cu132 (required for RTX 5070 sm_120) ───────────────────
run(f"{PIP} install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132")

# ── mmcv from source (must build with gcc-13, TORCH_CUDA_ARCH_LIST=9.0+PTX) ─
# Patch torch/utils/cpp_extension.py: change CUDA mismatch raise to warning
# Then:
#   cd ~/mmcv_build
#   MMCV_WITH_OPS=1 FORCE_CUDA=1 CUDA_HOME=~/cuda_home TORCH_CUDA_ARCH_LIST="9.0+PTX" \
#     CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13 \
#     pip install . --no-build-isolation --no-deps

# ── OpenMMLab stack ───────────────────────────────────────────────────────────
run(f"{PIP} install mmengine==0.10.4 mmdet==3.2.0 mmsegmentation==1.2.2 mmdet3d==1.4.0")

# ── Supporting packages ───────────────────────────────────────────────────────
run(f"{PIP} install 'numpy<2.0' 'matplotlib>=3.7,<3.9' ninja spconv-cu120")

# ── Verify ────────────────────────────────────────────────────────────────────
print("\n── Verification ─────────────────────────────────────────────────────────")
checks = {
    "torch+GPU": "import torch; print(torch.__version__, '| GPU:', torch.cuda.get_device_name(0))",
    "mmcv.ops":  "from mmcv.ops import box_iou_rotated; import torch; b=torch.ones(1,5,device='cuda'); print('OK')",
    "mmengine":  "import mmengine; print(mmengine.__version__)",
    "mmdet":     "import mmdet; print(mmdet.__version__)",
    "mmdet3d":   "import mmdet3d; print(mmdet3d.__version__)",
    "spconv":    "import spconv; print(spconv.__version__)",
}
for name, cmd in checks.items():
    r = subprocess.run(f'{PYTHON} -c "{cmd}"', shell=True, capture_output=True, text=True)
    status = "Passed" if r.returncode == 0 else "Failed"
    print(f"  {status} {name:<15} {r.stdout.strip() or r.stderr.strip()[-120:]}")

print(f"\nSetup complete. Run inference:\n  {PYTHON} /home/batashey/LCFusion_LT3D/inference.py")
