"""
DINO fine-tune — CURATED (multi-part) + ExcludingClassBalancedDataset(thr=0.30,
exclude barrier).

ABLATION: isolates whether OVERSAMPLING BARRIER ITSELF (not the multi-part
data, not the fusion calibration) causes barrier's fused-AP collapse seen in
dino_nuscenes_curated_cbd030.py (0.590 [Run C] -> 0.438). Diagnostic chain in
thesis_context.md (Aug 10 session) ruled out: label-geometry shift (checked,
no difference), 2D detection quality (checked, unchanged: AP 0.187 vs 0.180),
and fusion-time calibration (checked, ceiling ~0.453 via a c-sweep up to 2.0,
can't recover). The one variable never isolated: barrier's repeat factor was
~2x under ClassBalancedDataset(thr=0.30) on this curated data (see
verify_cbd_thresholds.py), which may compress its DINO confidence-score
distribution (matched 2D score dropped 0.634->0.454 despite match rate and
IoU both IMPROVING — a training-time calibration shift, not a detection
problem).

Identical to dino_nuscenes_curated_cbd030.py in every respect (same curated_v2
data — 62,720 images, parts 1,2,3,5,6; same oversample_thr=0.30; same 2 epochs;
same native multi-scale pipeline; filter_empty_gt=False) EXCEPT: uses
ExcludingClassBalancedDataset with exclude_categories=['barrier'], which
forces barrier's repeat factor to 1.0 (no duplication on barrier's account)
while every other class's repeat factor is computed exactly as before
(traffic_cone/trailer/construction_vehicle/bus still boosted 2x, bicycle/
motorcycle still 3x — unaffected by excluding barrier, since each image's
repeat factor is the max over its NON-excluded categories).

If barrier's fused AP recovers toward Run C's 0.590 under this ablation, it
confirms oversampling-of-barrier specifically is the cause (implying more
data + a barrier-aware balancing policy could safely combine). If it does
NOT recover, the multi-part data distribution itself (independent of
oversampling) is more likely the cause, and Track B (downloading further
parts) should stay paused regardless of balancing policy.

Train with:
  cd ~/mmdetection3d && PYTHONPATH=. \
  python ~/LCFusion_LT3D/train_dino.py \
    ~/LCFusion_LT3D/configs/dino_nuscenes_curated_cbd030_nobarrier.py
"""

_base_ = ['./dino_nuscenes_runA.py']   # multi-scale pipeline, 10-class head, COCO init

custom_imports = dict(imports=['excluding_class_balanced_dataset'],
                      allow_failed_imports=False)

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
DATA_ROOT = '/home/batashey/mmdetection3d/data/nuscenes_trainval/'

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ExcludingClassBalancedDataset',
        oversample_thr=0.30,
        exclude_categories=['barrier'],
        dataset=dict(
            type='CocoDataset',
            data_root=DATA_ROOT,
            metainfo=dict(classes=CLASSES),
            ann_file='nuscenes_2d_train_curated_v2.coco.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline={{_base_.train_pipeline}})))

max_epochs = 2
train_cfg = dict(max_epochs=max_epochs, val_interval=2)

work_dir = '/home/batashey/mmdetection3d/work_dirs/dino_nuscenes_curated_cbd030_nobarrier'
