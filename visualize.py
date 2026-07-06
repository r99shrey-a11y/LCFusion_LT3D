"""
Visualise LiDAR point clouds with ground-truth vs predicted 3D boxes.
Uses the SAME multi-sweep pipeline as the eval scripts so predictions match.
Saves top-down BEV images to ~/LCFusion_LT3D/viz/

Choose which model to visualise by setting MODEL below.

Usage: /home/batashey/miniconda3/envs/lcfusion/bin/python visualize.py
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
from utils import REPO, DATA_ROOTS, patch_cfg, torch

# ── CHOOSE MODEL AND DATASET HERE ────────────────────────────────────────────
MODEL   = "bevfusion"   # options: "centerpoint" or "bevfusion"
DATASET = "trainval"      # options: "mini" or "trainval"
N_SAMPLES = 4
# ──────────────────────────────────────────────────────────────────────────────

# Config + checkpoint for each model
MODELS = {
    "centerpoint": dict(
        cfg=os.path.join(REPO, "checkpoints/centerpoint",
            "centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"),
        ckpt=os.path.join(REPO, "checkpoints/centerpoint",
            "centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth"),
        title="CenterPoint",
    ),
    "bevfusion": dict(
        cfg=os.path.join(REPO, "projects/BEVFusion/configs",
            "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"),
        ckpt=os.path.join(REPO, "checkpoints/bevfusion",
            "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth"),
        title="BEVFusion (LiDAR-only)",
    ),
}

COLORS = {
    "car": "cyan", "truck": "orange", "construction_vehicle": "red",
    "bus": "yellow", "trailer": "magenta", "barrier": "brown",
    "motorcycle": "lime", "bicycle": "pink", "pedestrian": "white",
    "traffic_cone": "gold"
}

OUT_DIR = os.path.expanduser("~/LCFusion_LT3D/viz")
os.makedirs(OUT_DIR, exist_ok=True)

NUS_CLASSES = None  # populated in main() from the dataloader


def draw_bev_box(ax, box, color, label=None, alpha=1.0):
    """Draw one rotated box on a top-down (BEV) plot. box = [x,y,z,w,l,h,yaw]."""
    x, y, w, l, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[ l/2,  w/2], [-l/2,  w/2], [-l/2, -w/2], [ l/2, -w/2]])
    corners = (np.array([[c, -s], [s, c]]) @ corners.T).T + np.array([x, y])
    ax.add_patch(patches.Polygon(corners, closed=True, edgecolor=color,
                                 facecolor='none', linewidth=1.5, alpha=alpha))
    if label:
        ax.text(x, y, label[:3], color=color, fontsize=5, ha='center', va='center')


def plot_sample(points, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, idx, title):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('black')

    panels = [
        (axes[0], "Ground Truth", gt_boxes, gt_labels, None),
        (axes[1], f"{title} Predictions (score > 0.3)", pred_boxes, pred_labels, pred_scores),
    ]
    for ax, panel_title, boxes, labels, scores in panels:
        ax.set_facecolor('black')
        ax.set_title(panel_title, color='white', fontsize=11)
        ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
        ax.tick_params(colors='grey')

        m = (np.abs(points[:, 0]) < 50) & (np.abs(points[:, 1]) < 50)
        ax.scatter(points[m, 0], points[m, 1], s=0.1, c='white', alpha=0.3)

        for i in range(len(boxes)):
            cls = NUS_CLASSES[labels[i]] if labels[i] < len(NUS_CLASSES) else str(labels[i])
            a   = float(scores[i]) if scores is not None else 1.0
            draw_bev_box(ax, boxes[i], COLORS.get(cls, 'white'), label=cls, alpha=a)
        ax.plot(0, 0, 'r+', markersize=10)

    # Per-class count table (GT vs Pred) in the bottom-left corner
    from collections import Counter
    gt_counts   = Counter(NUS_CLASSES[l] for l in gt_labels   if l < len(NUS_CLASSES))
    pred_counts = Counter(NUS_CLASSES[l] for l in pred_labels if l < len(NUS_CLASSES))
    all_cls = [c for c in NUS_CLASSES if gt_counts.get(c, 0) or pred_counts.get(c, 0)]

    lines = [f"{'class':<14} GT  Pred"]
    for c in all_cls:
        lines.append(f"{c:<14}{gt_counts.get(c,0):>3}  {pred_counts.get(c,0):>4}")
    table_str = "\n".join(lines)

    axes[0].text(0.02, 0.02, table_str, transform=axes[0].transAxes,
                 color='lime', fontsize=7, family='monospace',
                 va='bottom', ha='left',
                 bbox=dict(facecolor='black', alpha=0.6, edgecolor='grey'))

    handles = [patches.Patch(edgecolor=c, facecolor='none', label=n) for n, c in COLORS.items()]
    axes[1].legend(handles=handles, loc='upper right', fontsize=6,
                   facecolor='black', labelcolor='white')

    out = os.path.join(OUT_DIR, f"{MODEL}_sample_{idx:02d}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {out}")


def main():
    cfg_info = MODELS[MODEL]
    sys.path.insert(0, REPO)
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.registry import init_default_scope

    init_default_scope('mmdet3d')
    cfg = patch_cfg(Config.fromfile(cfg_info['cfg']), cfg_info['ckpt'],
                    f"/tmp/viz_{MODEL}", dataset=DATASET)
    cfg.test_dataloader.batch_size = 1

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(cfg_info['ckpt'])
    model  = runner.model
    model.eval()
    loader = runner.test_dataloader

    # Read the authoritative class order from the dataset (avoids ordering bugs)
    global NUS_CLASSES
    NUS_CLASSES = list(loader.dataset.metainfo['classes'])
    print(f"Model: {cfg_info['title']}  |  Dataset: {DATASET}")
    print("Class order:", NUS_CLASSES)

    print(f"\nVisualising {N_SAMPLES} samples → {OUT_DIR}\n")
    for idx, data in enumerate(loader):
        if idx >= N_SAMPLES:
            break
        with torch.no_grad():
            out = model.test_step(data)[0]

        points = data['inputs']['points'][0].cpu().numpy()

        gt = out.eval_ann_info
        gt_boxes  = gt['gt_bboxes_3d'].tensor.numpy()
        gt_labels = np.array(gt['gt_labels_3d'])
        valid = gt_labels >= 0
        gt_boxes, gt_labels = gt_boxes[valid], gt_labels[valid]

        pred   = out.pred_instances_3d
        scores = pred.scores_3d.cpu().numpy()
        boxes  = pred.bboxes_3d.tensor.cpu().numpy()
        labels = pred.labels_3d.cpu().numpy()
        m = scores > 0.3
        boxes, labels, scores = boxes[m], labels[m], scores[m]

        print(f"Sample {idx}: GT={len(gt_boxes)} boxes, Pred={len(boxes)} boxes")
        plot_sample(points, gt_boxes, gt_labels, boxes, labels, scores, idx, cfg_info['title'])

    print(f"\nDone. Open the PNG files in {OUT_DIR}")


if __name__ == "__main__":
    main()
