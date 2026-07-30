"""
DINO fine-tune config — Run C (class-balanced, higher threshold).
Same as Run B but with oversample_thr=0.17 instead of 0.1, so that
construction_vehicle (11.2% img freq), barrier (14.9%), and traffic_cone
(16.0%) also fall below the cutoff and get oversampled, unlike Run B where
only trailer/bus/motorcycle/bicycle (<8.3%) were boosted.

Train with:
  cd ~/mmdetection3d && PYTHORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python ~/LCFusion_LT3D/train_dino.py \
    ~/LCFusion_LT3D/configs/dino_nuscenes_runC.py
"""

_base_ = ['./dino_nuscenes_runA.py']

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
DATA_ROOT = '/home/batashey/mmdetection3d/data/nuscenes_trainval/'

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

work_dir = '/home/batashey/mmdetection3d/work_dirs/dino_nuscenes_runC'
