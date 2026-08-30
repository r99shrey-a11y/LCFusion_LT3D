"""
Evaluate CenterPoint on nuScenes val set.
Choose the dataset by setting DATASET below.
"""

import os, sys
sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
from utils import REPO, patch_cfg, print_results

DATASET = "trainval"     # options: "mini" or "trainval"

sys.path.insert(0, REPO)
from mmengine.config import Config
from mmengine.runner import Runner

CFG_PATH = os.path.join(REPO, "checkpoints/centerpoint",
           "centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py")
CKPT     = os.path.join(REPO, "checkpoints/centerpoint",
           "centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth")

cfg     = patch_cfg(Config.fromfile(CFG_PATH), CKPT, "/tmp/cp_eval", dataset=DATASET)
metrics = Runner.from_cfg(cfg).test()
print_results("CenterPoint", metrics, dataset=DATASET)
