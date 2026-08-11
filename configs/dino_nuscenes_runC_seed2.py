"""
DINO fine-tune config — Run C, SEED VARIANT 2 (training-seed variance study).

See dino_nuscenes_runC_seed1.py for full rationale. Identical config, seed=2
instead of seed=1 — a second independent retrain to establish a 3-point
spread (original Run C [unfixed/lost seed] + seed1 + seed2) for
overall/tail/per-class mAP variance.

Train with:
  cd ~/mmdetection3d && PYTHONPATH=. \
  python ~/LCFusion_LT3D/train_dino.py \
    ~/LCFusion_LT3D/configs/dino_nuscenes_runC_seed2.py
"""

_base_ = ['./dino_nuscenes_runA.py']

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
DATA_ROOT = '/home/batashey/mmdetection3d/data/nuscenes_trainval/'

randomness = dict(seed=2)

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=0.17,
        dataset=dict(
            type='CocoDataset',
            data_root=DATA_ROOT,
            metainfo=dict(classes=CLASSES),
            ann_file='nuscenes_2d_train.coco.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline={{_base_.train_pipeline}})))

work_dir = '/home/batashey/mmdetection3d/work_dirs/dino_nuscenes_runC_seed2'
