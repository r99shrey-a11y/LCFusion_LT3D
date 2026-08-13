"""
Diagnostic: inspect DINO<->3D matching behaviour for a given DINO run,
per class, without changing the actual fusion output. Helps decide whether
low fusion gains are due to (a) poor 2D detection quality, (b) IoU threshold
being too strict, or (c) uncalibrated Bayesian weights.

Usage: /home/batashey/miniconda3/envs/lcfusion/bin/python diagnose_fusion.py
"""
import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
from utils import REPO, DATA_ROOTS, patch_cfg, torch
from late_fusion import (project_box, iou, CAMERAS, DINO_COVERED_ALL,
                         MODELS, IOU_THRESH as DEFAULT_IOU)

MODEL    = "bevfusion"
DATASET  = "trainval"
TARGET_CLASSES = ["truck", "trailer", "construction_vehicle", "barrier", "bus"]

# Test multiple IoU thresholds to see sensitivity
IOU_TEST_THRESHOLDS = [0.1, 0.3, 0.5]


def diagnose(dino_run):
    cfg_info = MODELS[MODEL]
    sys.path.insert(0, REPO)
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')

    cfg = patch_cfg(Config.fromfile(cfg_info['cfg']), cfg_info['ckpt'],
                    f"/tmp/diag_{MODEL}", dataset=DATASET)
    cfg.test_dataloader.batch_size = 1
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(cfg_info['ckpt'])
    model = runner.model
    model.eval()
    loader = runner.test_dataloader
    classes = list(loader.dataset.metainfo['classes'])

    dino = pickle.load(open(os.path.expanduser(
        f"~/LCFusion_LT3D/detections/dino_detections_{dino_run}.pkl"), "rb"))
    val_infos = pickle.load(open(
        os.path.join(DATA_ROOTS[DATASET], "nuscenes_infos_val.pkl"), "rb"))['data_list']

    # stats[cls][iou_thresh] = {n_3d, n_matched, iou_values, matched_2d_scores}
    stats = {c: {t: dict(n_3d=0, n_matched=0, ious=[], scores2d=[]) for t in IOU_TEST_THRESHOLDS}
              for c in TARGET_CLASSES}
    dino_raw_scores = {c: [] for c in TARGET_CLASSES}   # all DINO 2D scores for this class

    for data in loader:
        with torch.no_grad():
            out = model.test_step(data)[0]
        idx, info = out.metainfo['sample_idx'], val_infos[out.metainfo['sample_idx']]
        token = info['token']
        if token not in dino:
            continue

        pred = out.pred_instances_3d
        corners = pred.bboxes_3d.corners.cpu().numpy()
        scores  = pred.scores_3d.cpu().numpy()
        labels  = pred.labels_3d.cpu().numpy().astype(int)
        if len(corners) == 0:
            continue

        cam_calib = {cam: {"lidar2cam": np.array(info['images'][cam]['lidar2cam']),
                           "cam2img":   np.array(info['images'][cam]['cam2img'])}
                    for cam in CAMERAS}

        dino_by_cam = {c: [] for c in CAMERAS}
        for d in dino[token]:
            if d["nus_class"] in TARGET_CLASSES:
                dino_by_cam[d["cam"]].append(d)
                dino_raw_scores[d["nus_class"]].append(d["score"])

        for i in range(len(scores)):
            cls = classes[labels[i]]
            if cls not in TARGET_CLASSES or scores[i] < 0.05:
                continue
            for t in IOU_TEST_THRESHOLDS:
                stats[cls][t]['n_3d'] += 1

            # find best IoU match across all cameras (diagnostic only; does
            # not enforce one-to-one assignment like the real fusion does)
            for cam in CAMERAS:
                box2d = project_box(corners[i], cam_calib[cam]["lidar2cam"], cam_calib[cam]["cam2img"])
                if box2d is None:
                    continue
                best_iou, best_score = 0, 0
                for d in dino_by_cam[cam]:
                    if d["nus_class"] != cls:
                        continue
                    ov = iou(box2d, d["bbox"])
                    if ov > best_iou:
                        best_iou, best_score = ov, d["score"]
                if best_iou > 0:
                    for t in IOU_TEST_THRESHOLDS:
                        if best_iou >= t:
                            stats[cls][t]['n_matched'] += 1
                            stats[cls][t]['ious'].append(best_iou)
                            stats[cls][t]['scores2d'].append(best_score)

    return stats, dino_raw_scores


def main():
    for run in ["oversampled", "curated_cbd030"]:
        print(f"\n{'='*70}\nDINO {run}\n{'='*70}")
        stats, raw = diagnose(run)
        for cls in TARGET_CLASSES:
            n_raw = len(raw[cls])
            avg_raw_score = np.mean(raw[cls]) if n_raw else 0
            print(f"\n  Class: {cls}")
            print(f"    Total DINO 2D detections (all cams, val set): {n_raw}"
                  f"  (avg score {avg_raw_score:.3f})")
            for t in IOU_TEST_THRESHOLDS:
                s = stats[cls][t]
                rate = s['n_matched'] / s['n_3d'] * 100 if s['n_3d'] else 0
                avg_iou = np.mean(s['ious']) if s['ious'] else 0
                avg_s2d = np.mean(s['scores2d']) if s['scores2d'] else 0
                print(f"    IoU>={t:.1f}: {s['n_matched']:>4}/{s['n_3d']:<4} 3D boxes matched "
                      f"({rate:5.1f}%)  avg IoU={avg_iou:.2f}  avg matched 2D score={avg_s2d:.3f}")


if __name__ == "__main__":
    main()
