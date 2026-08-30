"""
Generate the full candidate pool of 2D COCO-format annotations for DINO
fine-tuning, by projecting nuScenes 3D boxes onto the 6 camera images across
all downloaded trainval parts.

Outputs nuscenes_2d_train_pool.coco.json — every train-scene camera keyframe
with its projected 2D boxes. No filtering is applied here; image-selection
(curating) is done separately in make_curated_v2.py for fast iteration.
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
OUT     = os.path.join(ROOT, "nuscenes_2d_train_pool.coco.json")
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
VIS     = ['', '1', '2', '3', '4']   # all visibility levels

nusc = NuScenes(version="v1.0-trainval", dataroot=ROOT, verbose=False)
name_to_scene = {s['name']: s for s in nusc.scene}


def scene_present(scene):
    """A scene is available if every keyframe's CAM_FRONT jpg is extracted."""
    tok = nusc.get('scene', scene['token'])['first_sample_token']
    while tok:
        s = nusc.get('sample', tok)
        sd = nusc.get('sample_data', s['data']['CAM_FRONT'])
        if not os.path.exists(os.path.join(ROOT, sd['filename'])):
            return False
        tok = s['next']
    return True


def main():
    categories = [dict(id=i, name=n) for i, n in enumerate(nus_categories)]
    coco = dict(images=[], annotations=[], categories=categories)
    ann_id = 0
    present, absent = 0, 0

    for name in mmengine.track_iter_progress(list(splits.train)):
        if name not in name_to_scene:
            continue
        if not scene_present(name_to_scene[name]):
            absent += 1
            continue
        present += 1
        tok = nusc.get('scene', name_to_scene[name]['token'])['first_sample_token']
        while tok:
            sample = nusc.get('sample', tok)
            for cam in CAMERAS:
                sd_token = sample['data'][cam]
                sd = nusc.get('sample_data', sd_token)
                coco['images'].append(dict(
                    file_name=sd['filename'], id=sd_token,
                    width=sd['width'], height=sd['height']))
                recs = get_2d_boxes(nusc, sd_token, visibilities=VIS, mono3d=False)
                for r in recs:
                    if r is None:
                        continue
                    coco['annotations'].append(dict(
                        id=ann_id, image_id=sd_token,
                        category_id=r['category_id'],
                        bbox=r['bbox'], area=r['area'],
                        iscrowd=0, segmentation=[]))
                    ann_id += 1
            tok = sample['next']

    json.dump(coco, open(OUT, 'w'))
    print(f"\nscenes present={present} absent={absent}")
    print(f"wrote {len(coco['images'])} images, {len(coco['annotations'])} anns → {OUT}")


if __name__ == "__main__":
    main()
