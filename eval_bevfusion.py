"""
Evaluate BEVFusion (LiDAR-only) on nuScenes val set.
Choose the dataset by setting DATASET below.
Usage: /home/batashey/miniconda3/envs/lcfusion/bin/python eval_bevfusion.py
"""

import os, sys
sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
from utils import REPO, patch_cfg, print_results

# ── CHOOSE DATASET HERE ───────────────────────────────────────────────────────
DATASET = "trainval"     # options: "mini" or "trainval"
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, REPO)  # needed for BEVFusion custom modules
from mmengine.config import Config
from mmengine.runner import Runner

CFG_PATH = os.path.join(REPO, "projects/BEVFusion/configs",
           "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
CKPT     = os.path.join(REPO, "checkpoints/bevfusion",
           "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth")

cfg     = patch_cfg(Config.fromfile(CFG_PATH), CKPT, "/tmp/bev_eval", dataset=DATASET)
metrics = Runner.from_cfg(cfg).test()
print_results("BEVFusion (LiDAR-only)", metrics, dataset=DATASET)
