"""
DINO fine-tune config — Run B (class-balanced).
Same as Run A but wraps the train dataset in ClassBalancedDataset, which
oversamples images containing rare classes (bus, trailer, construction_vehicle
etc.) so the model sees them more often.

Train with:
  cd ~/mmdetection3d && PYTHONPATH=. \
  python ~/mmdetection3d/tools/train.py \
    ~/LCFusion_LT3D/configs/dino_nuscenes_runB.py
"""

_base_ = ['./dino_nuscenes_runA.py']

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
DATA_ROOT = '/home/batashey/mmdetection3d/data/nuscenes_trainval/'

# Wrap the train dataset: oversample images whose rarest class appears in
# fewer than `oversample_thr` fraction of images.
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=0.1,
        dataset=dict(
            type='CocoDataset',
            data_root=DATA_ROOT,
            metainfo=dict(classes=CLASSES),
            ann_file='nuscenes_2d_train.coco.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline={{_base_.train_pipeline}})))

work_dir = '/home/batashey/mmdetection3d/work_dirs/dino_nuscenes_runB'
