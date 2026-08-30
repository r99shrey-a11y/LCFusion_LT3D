"""
Generate 2D COCO-format annotations for DINO fine-tuning by projecting
nuScenes 3D boxes onto the 6 camera images.

Produces (in data/nuscenes_trainval/):
  nuscenes_2d_train.coco.json   — present train scenes in part 1
  nuscenes_2d_val.coco.json     — present val scenes in part 1

Reuses get_2d_boxes / nus_categories from mmdet3d's nuscenes_converter.

"""
import os, sys, json
sys.path.insert(0, os.path.expanduser("~/mmdetection3d"))

from mmengine.registry import init_default_scope
init_default_scope('mmdet3d')

import mmengine
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tools.dataset_converters.nuscenes_converter import get_2d_boxes, nus_categories

ROOT    = "data/nuscenes_trainval"
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
VIS     = ['', '1', '2', '3', '4']   # all visibility levels

nusc = NuScenes(version="v1.0-trainval", dataroot=ROOT, verbose=False)
name_to_scene = {s['name']: s for s in nusc.scene}

# checks if for each sample in the scene its lidar file exists otherwise skip the scene.
def scene_present(scene):
    tok = nusc.get('scene', scene['token'])['first_sample_token']
    while tok:
        s = nusc.get('sample', tok)
        sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
        if not os.path.exists(os.path.join(ROOT, sd['filename'])):
            return False
        tok = s['next']
    return True


def build_coco(scene_names, out_name):
    categories = [dict(id=i, name=n) for i, n in enumerate(nus_categories)]
    coco = dict(images=[], annotations=[], categories=categories)
    ann_id = 0

    for name in mmengine.track_iter_progress(scene_names):
        if name not in name_to_scene or not scene_present(name_to_scene[name]):
            continue
        tok = nusc.get('scene', name_to_scene[name]['token'])['first_sample_token']
        while tok:
            sample = nusc.get('sample', tok)
            for cam in CAMERAS:
                sd_token = sample['data'][cam]
                sd = nusc.get('sample_data', sd_token)
                coco['images'].append(dict(
                    file_name=sd['filename'],        
                    id=sd_token,
                    width=sd['width'],
                    height=sd['height'],
                ))
                recs = get_2d_boxes(nusc, sd_token, visibilities=VIS, mono3d=False)
                for r in recs:
                    if r is None:
                        continue
                    coco['annotations'].append(dict(
                        id=ann_id,
                        image_id=sd_token,
                        category_id=r['category_id'],
                        bbox=r['bbox'],            # [x, y, w, h]
                        area=r['area'],
                        iscrowd=0,
                        segmentation=[],
                    ))
                    ann_id += 1
            tok = sample['next']

    out_path = os.path.join(ROOT, out_name)
    json.dump(coco, open(out_path, 'w'))
    print(f"\n{out_name}: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations → {out_path}")


if __name__ == "__main__":
    print("Building val COCO annotations...")
    build_coco(splits.val,   "nuscenes_2d_val.coco.json")
    print("Building train COCO annotations...")
    build_coco(splits.train, "nuscenes_2d_train.coco.json")
