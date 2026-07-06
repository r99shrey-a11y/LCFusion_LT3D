"""
Generate nuscenes_infos_val.pkl for the trainval data actually present (part 1).
Filters val scenes to those whose LiDAR files exist on disk, then builds infos.

Usage: cd ~/mmdetection3d && PYTHONPATH=. python ~/LCFusion_LT3D/make_trainval_infos.py
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/mmdetection3d"))

from mmengine.registry import init_default_scope
init_default_scope('mmdet3d')

import mmengine
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tools.dataset_converters import nuscenes_converter
import tools.dataset_converters.update_infos_to_v2 as update_mod
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos

ROOT   = "data/nuscenes_trainval"
PREFIX = "nuscenes"

# The converter validates every sample's file before checking its split,
# which would crash on scenes outside part 1. Make the check a no-op.
mmengine.check_file_exist = lambda *a, **k: None
nuscenes_converter.mmengine.check_file_exist = lambda *a, **k: None

# update_nuscenes_infos hardcodes dataroot='./data/nuscenes'. Force ours.
_OrigNuSc = update_mod.NuScenes
update_mod.NuScenes = lambda version, dataroot, verbose: _OrigNuSc(
    version=version, dataroot=ROOT, verbose=verbose)

# ── Load nuScenes metadata ────────────────────────────────────────────────────
nusc = NuScenes(version="v1.0-trainval", dataroot=ROOT, verbose=False)


def scene_files_exist(scene):
    """True if every keyframe LiDAR file of the scene exists on disk."""
    token = nusc.get('scene', scene['token'])['first_sample_token']
    while token:
        sample = nusc.get('sample', token)
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        if not os.path.exists(os.path.join(ROOT, sd['filename'])):
            return False
        token = sample['next']
    return True


# ── Filter the official val split to scenes present in part 1 ─────────────────
name_to_scene = {s['name']: s for s in nusc.scene}
present_val = [name for name in splits.val
               if name in name_to_scene and scene_files_exist(name_to_scene[name])]

print(f"Official val scenes: {len(splits.val)}")
print(f"Val scenes present in part 1: {len(present_val)}")

# The converter assigns a sample to TRAIN if its scene is in train_scenes,
# otherwise to VAL. So put every non-present-val scene into train (discarded),
# leaving only the present val scenes in the val infos.
all_names = [s['name'] for s in nusc.scene]
splits.val   = present_val
splits.train = [n for n in all_names if n not in set(present_val)]

# ── Build infos (only present val scenes end up in the val pkl) ───────────────
nuscenes_converter.create_nuscenes_infos(ROOT, PREFIX, version="v1.0-trainval", max_sweeps=10)

val_path = os.path.join(ROOT, f"{PREFIX}_infos_val.pkl")
update_pkl_infos('nuscenes', out_dir=ROOT, pkl_path=val_path)
print("\nDone. Val info pkl ready at", val_path)
