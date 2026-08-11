"""Diagnostic: per-class 2D AP for barrier (and neighbors it might be
confused with), comparing Run C's checkpoint vs curated_cbd030's checkpoint
on the SAME frozen val set. Uses CocoMetric(classwise=True)."""
import sys, os, torch
sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
_orig = torch.load
torch.load = lambda *a, **k: (k.setdefault('weights_only', False), _orig(*a, **k))[1]

from mmengine.config import Config
from mmengine.runner import Runner

RUNS = {
    "RunC (part-1, thr=0.17)": dict(
        cfg="~/LCFusion_LT3D/configs/dino_nuscenes_runC.py",
        ckpt="~/mmdetection3d/work_dirs/dino_nuscenes_runC/epoch_4.pth"),
    "Curated+CBD030 (5 parts, thr=0.30)": dict(
        cfg="~/LCFusion_LT3D/configs/dino_nuscenes_curated_cbd030.py",
        ckpt="~/mmdetection3d/work_dirs/dino_nuscenes_curated_cbd030/epoch_2.pth"),
}

for name, paths in RUNS.items():
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    cfg = Config.fromfile(os.path.expanduser(paths["cfg"]))
    cfg.launcher = 'none'
    cfg.load_from = os.path.expanduser(paths["ckpt"])
    cfg.test_evaluator.classwise = True
    cfg.test_evaluator.pop('format_only', None)
    runner = Runner.from_cfg(cfg)
    runner.test()
