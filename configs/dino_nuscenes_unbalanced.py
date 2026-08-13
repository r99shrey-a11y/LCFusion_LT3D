"""
DINO fine-tune config — DINO fine-tune — UNBALANCED (natural class distribution).
Fine-tunes COCO-pretrained DINO on nuScenes 2D annotations (10 classes).

Train with:
  cd ~/mmdetection3d && PYTHONPATH=. \
  python ~/mmdetection3d/tools/train.py \
    ~/LCFusion_LT3D/configs/dino_nuscenes_unbalanced.py
"""

_base_ = ['/home/batashey/mmdetection3d/checkpoints/dino/dino-4scale_r50_8xb2-12e_coco.py']

# nuScenes 10 detection classes (COCO json category order)
CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')

DATA_ROOT = '/home/batashey/mmdetection3d/data/nuscenes_trainval/'

# Start from the COCO-pretrained DINO checkpoint (fine-tune, not from scratch)
load_from = ('/home/batashey/mmdetection3d/checkpoints/dino/'
             'dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth')

# 10-class detection head
model = dict(bbox_head=dict(num_classes=10))

metainfo = dict(classes=CLASSES)

train_dataloader = dict(
    batch_size=1,               # RTX 5070 12GB — DINO is memory heavy
    num_workers=2,
    dataset=dict(
        data_root=DATA_ROOT,
        metainfo=metainfo,
        ann_file='nuscenes_2d_train.coco.json',
        data_prefix=dict(img='')))   # file_name already = samples/CAM_.../xxx.jpg

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

# Shorter schedule for fine-tuning on a single GPU (from COCO-pretrained init)
max_epochs = 4
train_cfg = dict(max_epochs=max_epochs, val_interval=2)

# Lower LR for fine-tuning
optim_wrapper = dict(optimizer=dict(lr=0.0001))

work_dir = '/home/batashey/mmdetection3d/work_dirs/dino_nuscenes_unbalanced'
