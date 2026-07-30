"""
DINO fine-tune config — CURATED (v2).
Fine-tunes COCO-pretrained DINO on the curated nuScenes 2D train set built from
parts 1+2+3: every tail-containing and bicycle-containing image + empties (kept
as background negatives), with all pure car/pedestrian-only images dropped.
(36,588 images; 43.1% tail instances.) Val set is the frozen part-1 set.

Natural sampling (no ClassBalancedDataset) — the data is already rebalanced by
curation. A separate config can stack oversampling on top if desired.

Train with:
  cd ~/mmdetection3d && PYTHONPATH=. \
  python ~/LCFusion_LT3D/train_dino.py \
    ~/LCFusion_LT3D/configs/dino_nuscenes_curated.py
"""

_base_ = ['/home/batashey/mmdetection3d/checkpoints/dino/dino-4scale_r50_8xb2-12e_coco.py']

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')

DATA_ROOT = '/home/batashey/mmdetection3d/data/nuscenes_trainval/'

load_from = ('/home/batashey/mmdetection3d/checkpoints/dino/'
             'dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth')

model = dict(bbox_head=dict(num_classes=10))

metainfo = dict(classes=CLASSES)

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=DATA_ROOT,
        metainfo=metainfo,
        ann_file='nuscenes_2d_train_curated_v2.coco.json',
        data_prefix=dict(img='')))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        data_root=DATA_ROOT,
        metainfo=metainfo,
        ann_file='nuscenes_2d_val.coco.json',
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=DATA_ROOT + 'nuscenes_2d_val.coco.json')
test_evaluator = val_evaluator

max_epochs = 4
train_cfg = dict(max_epochs=max_epochs, val_interval=2)

optim_wrapper = dict(optimizer=dict(lr=0.0001))

work_dir = '/home/batashey/mmdetection3d/work_dirs/dino_nuscenes_curated'
