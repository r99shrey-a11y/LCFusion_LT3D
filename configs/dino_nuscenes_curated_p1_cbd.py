"""
DINO fine-tune — CURATED(part-1, ratio=0.10) + ClassBalancedDataset(thr=0.17).

Direct, same-data comparison against Run C: same part-1 scenes, same
ClassBalancedDataset oversampling settings and multi-scale pipeline as Run C
(inherited via dino_nuscenes_runA.py's train_pipeline), the ONLY difference
is the base annotation file — curated (drop most car/ped-only images, ratio
0.10) instead of the raw part-1 train set. Isolates: does curating away excess
head-only images, before oversampling, improve on oversampling alone?

Train with:
  cd ~/mmdetection3d && PYTHONPATH=. \
  python ~/LCFusion_LT3D/train_dino.py \
    ~/LCFusion_LT3D/configs/dino_nuscenes_curated_p1_cbd.py
"""

_base_ = ['./dino_nuscenes_runA.py']   # same multi-scale pipeline as Run C

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
DATA_ROOT = '/home/batashey/mmdetection3d/data/nuscenes_trainval/'

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=0.17,        # same threshold as Run C
        dataset=dict(
            type='CocoDataset',
            data_root=DATA_ROOT,
            metainfo=dict(classes=CLASSES),
            ann_file='nuscenes_2d_train_curated_p1_r010.coco.json',  # ONLY difference vs Run C
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline={{_base_.train_pipeline}})))

work_dir = '/home/batashey/mmdetection3d/work_dirs/dino_nuscenes_curated_p1_cbd'
