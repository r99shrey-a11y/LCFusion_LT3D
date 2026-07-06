"""Shared utilities for all scripts."""

import os, sys, json, torch
import numpy as np

# Paths
REPO = os.path.expanduser("~/mmdetection3d")

# Two datasets are supported. Pick one via the `dataset` arg to patch_cfg().
DATA_ROOTS = {
    "mini":     os.path.expanduser("~/mmdetection3d/data/nuscenes"),
    "trainval": os.path.expanduser("~/mmdetection3d/data/nuscenes_trainval"),
}
VERSIONS = {"mini": "v1.0-mini", "trainval": "v1.0-trainval"}

# Default (kept for backwards compatibility with older scripts)
DATA_ROOT = DATA_ROOTS["mini"]

# nuScenes class groups
NUSCENES_CLASSES = ["car", "truck", "trailer", "bus", "construction_vehicle",
                    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"]
HEAD_CLASSES = {"car", "pedestrian", "bicycle"}
TAIL_CLASSES = {"truck", "bus", "trailer", "construction_vehicle",
                "motorcycle", "traffic_cone", "barrier"}

# Fix torch.load for PyTorch 2.13+ (mmengine checkpoints need weights_only=False)
_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault('weights_only', False)
    return _orig_load(*a, **k)
torch.load = _patched_load


def patch_cfg(cfg, checkpoint_path, work_dir, dataset="mini"):
    """Point config to the chosen val set and checkpoint.

    dataset: "mini" (81 samples) or "trainval" (part-1 subset, 914 samples).
    """
    data_root = DATA_ROOTS[dataset] + "/"
    version   = VERSIONS[dataset]

    cfg.test_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.ann_file  = "nuscenes_infos_val.pkl"
    cfg.test_dataloader.dataset.metainfo  = dict(
        classes=cfg.test_dataloader.dataset.metainfo['classes'],
        version=version,
    )
    cfg.test_evaluator.ann_file  = os.path.join(DATA_ROOTS[dataset], "nuscenes_infos_val.pkl")
    cfg.test_evaluator.data_root = data_root
    cfg.test_evaluator.pop('version', None)
    cfg.work_dir  = work_dir
    cfg.load_from = checkpoint_path

    if dataset == "trainval":
        _patch_eval_split_to_subset()
    return cfg


def _patch_eval_split_to_subset():
    """Restrict the nuScenes 'val' eval split to the scenes we actually have
    (part 1). Without this the official evaluator expects all 150 val scenes."""
    scenes_file = os.path.expanduser("~/LCFusion_LT3D/trainval_val_scenes.json")
    present = json.load(open(scenes_file))

    import nuscenes.eval.common.loaders as loaders
    _orig = loaders.create_splits_scenes
    def _patched(*a, **k):
        splits = _orig(*a, **k)
        splits['val'] = present     # override official 150-scene val list
        return splits
    loaders.create_splits_scenes = _patched


def print_results(model_name, metrics, dataset="mini"):
    """Print per-class AP with head/tail/overall summary."""
    PREFIX = "NuScenes metric/pred_instances_3d_NuScenes/"
    DISTS  = ["AP_dist_0.5", "AP_dist_1.0", "AP_dist_2.0", "AP_dist_4.0"]
    split_name = "trainval part-1 val" if dataset == "trainval" else "mini val"

    print(f"\n{'='*55}")
    print(f"{model_name} — Per-Class AP  (nuScenes {split_name})")
    print(f"{'='*55}")

    head_aps, tail_aps = [], []
    for cls in NUSCENES_CLASSES:
        ap  = float(np.mean([metrics[f"{PREFIX}{cls}_{d}"] for d in DISTS]))
        tag = "HEAD" if cls in HEAD_CLASSES else "TAIL"
        print(f"  {tag}  {cls:<25}  AP = {ap:.3f}")
        (head_aps if cls in HEAD_CLASSES else tail_aps).append(ap)

    print(f"\n  Head mAP    : {np.mean(head_aps):.3f}")
    print(f"  Tail mAP    : {np.mean(tail_aps):.3f}")
    print(f"  Overall mAP : {metrics[f'{PREFIX}mAP']:.3f}")
    print(f"  NDS         : {metrics[f'{PREFIX}NDS']:.3f}")
