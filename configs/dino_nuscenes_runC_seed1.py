"""
DINO fine-tune config — Run C, SEED VARIANT 1 (training-seed variance study).

Identical to dino_nuscenes_runC.py in every respect (same part-1 data,
ClassBalancedDataset oversample_thr=0.17, same multi-scale pipeline, same
4 epochs) EXCEPT a fixed random seed is set via mmengine's `randomness`
config field. The original Run C never set a seed (mmengine auto-generates
one via sync_random_seed() when unset, logged only to that run's own stdout
which was not preserved) — so this is not a reproduction of the exact
original run, but an independent same-config retrain to measure how much
overall/tail/per-class mAP varies run-to-run purely from training-time
randomness (weight init of the new 10-class head, data shuffling order,
augmentation choices), holding data/config/architecture fixed.

`deterministic=False` (default) is kept intentionally: we want the realistic
run-to-run spread including cuDNN's non-deterministic algorithm selection,
not artificial bit-for-bit reproducibility.

Train with:
  cd ~/mmdetection3d && PYTHONPATH=. \
  python ~/LCFusion_LT3D/train_dino.py \
    ~/LCFusion_LT3D/configs/dino_nuscenes_runC_seed1.py
"""

_base_ = ['./dino_nuscenes_runA.py']

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
DATA_ROOT = '/home/batashey/mmdetection3d/data/nuscenes_trainval/'

randomness = dict(seed=1)

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

work_dir = '/home/batashey/mmdetection3d/work_dirs/dino_nuscenes_runC_seed1'
