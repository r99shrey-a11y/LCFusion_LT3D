"""
Evaluate BEVFusion (camera+LiDAR) on nuScenes val set — no DINO fusion (baseline).
"""

import os, sys
sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
from utils import REPO, patch_cfg, print_results

DATASET = "trainval"     # "mini" or "trainval"

sys.path.insert(0, REPO)  # needed for BEVFusion custom modules
from mmengine.config import Config
from mmengine.runner import Runner

CFG_PATH = os.path.join(REPO, "projects/BEVFusion/configs",
           "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
CKPT     = os.path.join(REPO, "checkpoints/bevfusion",
           "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth")

cfg     = patch_cfg(Config.fromfile(CFG_PATH), CKPT, "/tmp/bev_cam_eval", dataset=DATASET)
metrics = Runner.from_cfg(cfg).test()
print_results("BEVFusion (camera+LiDAR)", metrics, dataset=DATASET)
