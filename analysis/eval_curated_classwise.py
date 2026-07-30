"""Per-class 2D AP for the curated DINO checkpoint on the frozen val set.
Runs one val pass with CocoMetric(classwise=True)."""
import sys, os, torch
sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
_orig = torch.load
torch.load = lambda *a, **k: (k.setdefault('weights_only', False), _orig(*a, **k))[1]

from mmengine.config import Config
from mmengine.runner import Runner

CFG  = os.path.expanduser("~/LCFusion_LT3D/configs/dino_nuscenes_curated.py")
CKPT = os.path.expanduser("~/mmdetection3d/work_dirs/dino_nuscenes_curated/epoch_4.pth")

cfg = Config.fromfile(CFG)
cfg.launcher = 'none'
cfg.load_from = CKPT
cfg.test_evaluator.classwise = True
cfg.test_evaluator.pop('format_only', None)
runner = Runner.from_cfg(cfg)
runner.test()
