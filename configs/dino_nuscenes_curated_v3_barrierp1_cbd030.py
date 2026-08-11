"""
DINO fine-tune — CURATED v3 (barrier volume partially restricted to part-1)
+ ClassBalancedDataset(oversample_thr=0.30).

ABLATION (Option B, scoped-down per user decision — see thesis_context.md):
tests whether barrier's fused-AP collapse is driven by the raw VOLUME/
diversity of its multi-part training exposure, as opposed to oversampling
repetition (already ruled out by dino_nuscenes_curated_cbd030_nobarrier.py:
excluding barrier from oversampling gave 0.449, no real recovery from 0.438).

Data: nuscenes_2d_train_curated_v3_barrierp1.coco.json
(analysis/make_curated_v3_barrierp1.py) — built from the curated_v2 policy
(keep every tail/bicycle/empty image, drop pure car/ped-only) PLUS: images
whose ONLY tail class is barrier are kept only if sourced from part-1,
dropped if sourced from parts 2/3/5/6. Images containing barrier AND another
tail class are ALWAYS kept (dropping them would also remove the other tail
class's signal) — this is why barrier's volume could not be fully equalized
to Run C's 18,170 boxes: 85% of barrier's multi-part excess (48,964 of 57,359
extra boxes) comes from images where barrier co-occurs with another tail
class. Only the "barrier-only" images (8,395 boxes) were removable without
an annotation-deletion approach, which was explicitly rejected (missing-
annotation risk: an unlabeled visible barrier teaches the model to treat it
as background, the same risk flagged for "instance-drop by deletion" earlier
in this project).

Result: barrier volume reduced from curated_cbd030's 75,529 -> 67,134 boxes
(a partial, ~11% reduction, NOT equalized to Run C's 18,170) while every
other tail class keeps its full multi-part volume unchanged (construction_
vehicle 9,803, trailer 19,413, bus 7,139, etc. -- identical to
dino_nuscenes_curated_cbd030.py). This tests the directional hypothesis
(does less barrier exposure move barrier's AP at all) without claiming an
exact volume-matched comparison to Run C.

Otherwise identical to dino_nuscenes_curated_cbd030.py: same oversample_thr
=0.30 (verified appropriate for this data's frequencies via
verify_cbd_thresholds.py — barrier freq shifts from 21.6% to 19.1% on this
slightly smaller set, r@0.30 becomes 1.25 rather than 1.27, negligible
change to the ceil-based repeat factor), same 2 epochs, same native
multi-scale pipeline, filter_empty_gt=False (curation keeps 10,862 empty
images as negatives).

Train with:
  cd ~/mmdetection3d && PYTHONPATH=. \
  python ~/LCFusion_LT3D/train_dino.py \
    ~/LCFusion_LT3D/configs/dino_nuscenes_curated_v3_barrierp1_cbd030.py
"""

_base_ = ['./dino_nuscenes_runA.py']   # multi-scale pipeline, 10-class head, COCO init

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
DATA_ROOT = '/home/batashey/mmdetection3d/data/nuscenes_trainval/'

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=0.30,
        dataset=dict(
            type='CocoDataset',
            data_root=DATA_ROOT,
            metainfo=dict(classes=CLASSES),
            ann_file='nuscenes_2d_train_curated_v3_barrierp1.coco.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline={{_base_.train_pipeline}})))

max_epochs = 2
train_cfg = dict(max_epochs=max_epochs, val_interval=2)

work_dir = '/home/batashey/mmdetection3d/work_dirs/dino_nuscenes_curated_v3_barrierp1_cbd030'
