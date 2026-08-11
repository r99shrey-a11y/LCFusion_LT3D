"""
DINO fine-tune — CURATED (multi-part) + ClassBalancedDataset(oversample_thr=0.30).

Trains on the curated pool built from all downloaded trainval parts
(1, 2, 3, 5, 6): `nuscenes_2d_train_curated_v2.coco.json` — 62,720 images
(508,766 boxes, 44.2% tail), keeping every tail-containing and every
bicycle-containing image plus 10,862 empty images as background negatives,
with all pure car/pedestrian-only images dropped.

WHY A RE-TUNED THRESHOLD (Run C used 0.17):
Curation raised per-class image frequencies, which silently disabled
oversampling for the two most numerous tail classes. mmengine applies
math.ceil() to each image's repeat factor (dataset_wrapper.py:
`[dataset_index] * math.ceil(repeat_factor)`), so only the integer bucket
matters. Verified on this exact file via analysis/verify_cbd_thresholds.py:

  class                 imgfreq   r@0.17    r@0.30    r@0.35
  car                     61.7%   1.00->1   1.00->1   1.00->1
  truck                   50.3%   1.00->1   1.00->1   1.00->1
  pedestrian              34.1%   1.00->1   1.00->1   1.01->2  <-- diluted
  traffic_cone            23.7%   1.00->1   1.12->2   1.21->2
  barrier                 21.6%   1.00->1   1.18->2   1.27->2
  trailer                 15.8%   1.04->2   1.38->2   1.49->2
  construction_vehicle    11.0%   1.24->2   1.65->2   1.78->2
  bus                      9.4%   1.35->2   1.79->2   1.93->2
  bicycle                  6.7%   1.60->2   2.12->3   2.29->3
  motorcycle               6.2%   1.65->2   2.20->3   2.37->3

At 0.17 barrier and traffic_cone sit ABOVE the cutoff and get zero
oversampling — the same failure mode that made Run B (thr=0.1) exclude
construction_vehicle. barrier is the highest-variance tail class across all
runs (0.362-0.619) and the one where late fusion helps most (BEVFusion alone
0.446 -> best fused 0.619), so boosting it is the highest-leverage change.

0.30 is chosen over 0.35 because under ceil() they give IDENTICAL buckets for
all seven tail classes, but 0.35 also pushes pedestrian to 2 copies —
duplicating 4,606 images purely to boost an already-saturated head class
(pedestrian AP 0.899). The valid band keeping car/truck/pedestrian at 1 copy
while giving bicycle/motorcycle 3 copies is 0.249 < thr <= 0.341; 0.30 sits
safely mid-band rather than on the 0.341 boundary.

Effective dataset: 109,167 images/epoch (1.74x the 62,720 base).
Copy distribution: 23,869 images 1x, 31,255 2x, 7,596 3x.

filter_empty_gt=False is REQUIRED: the curation deliberately keeps the 10,862
empty images as background negatives, and the frequencies above are computed
with them in the denominator. Setting True would drop them, shrink the base to
51,858, shift every frequency upward and invalidate the chosen threshold.
(mmengine assigns repeat_factor=1.0 to images with no categories, so empties
are never duplicated.)

2 epochs (not 4): at 109,167 img/epoch this is ~218k optimizer steps, well
above the 4-epoch curated run's 146k. The part-1 curated+oversampling run
showed 2D val mAP degrading from epoch 2 (0.231) to epoch 4 (0.215).

Native DINO multi-scale pipeline is kept (inherited via _base_.train_pipeline).
No copy-paste, no fixed-scale Resize — the earlier copy-paste run had to force
a fixed 1067x600 scale, which cut small-object AP (mAP_s 0.033 -> 0.024) and is
the leading suspect for its barrier collapse to 0.362.

Train with:
  cd ~/mmdetection3d && PYTHONPATH=. \
  python ~/LCFusion_LT3D/train_dino.py \
    ~/LCFusion_LT3D/configs/dino_nuscenes_curated_cbd030.py
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
            ann_file='nuscenes_2d_train_curated_v2.coco.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline={{_base_.train_pipeline}})))

max_epochs = 2
train_cfg = dict(max_epochs=max_epochs, val_interval=2)

work_dir = '/home/batashey/mmdetection3d/work_dirs/dino_nuscenes_curated_cbd030'
