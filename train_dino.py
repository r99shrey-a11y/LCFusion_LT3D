"""
DINO fine-tune launcher. Patches torch.load (PyTorch 2.13 weights_only issue)
then trains using the given config.

Usage:
  cd ~/mmdetection3d && PYTHONPATH=. \
    python ~/LCFusion_LT3D/train_dino.py <config.py>

Example:
  python ~/LCFusion_LT3D/train_dino.py ~/LCFusion_LT3D/configs/dino_nuscenes_runA.py
"""
import sys, os, torch

# Make ~/LCFusion_LT3D importable so custom config imports can be found.
sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))

# Patch torch.load for PyTorch 2.13+ (mmengine/DINO checkpoints need weights_only=False)
_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault('weights_only', False)
    return _orig_load(*a, **k)
torch.load = _patched_load

from mmengine.config import Config
from mmengine.runner import Runner


def main(config_path):
    cfg = Config.fromfile(config_path)
    cfg.launcher = 'none'
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: python train_dino.py <config.py>"
    main(sys.argv[1])
