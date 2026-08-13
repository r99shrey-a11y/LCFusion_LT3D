"""
Visualise FUSED 3D+2D predictions (BEV) for two configs side by side:
  CenterPoint + DINO(B)  vs  BEVFusion + DINO (oversampled)
Reuses the exact fusion logic from late_fusion.py (fuse_sample, bayes_fuse, CAL)
and the exact plotting style from visualize.py — no fusion math duplicated.

Saves 3-panel PNGs (GT | CenterPoint+DINO(B) | BEVFusion+DINO(C)) to
~/LCFusion_LT3D/viz/fused_sample_XX.png

Usage: /home/batashey/miniconda3/envs/lcfusion/bin/python visualize_fused.py
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
from utils import REPO, DATA_ROOTS, patch_cfg, torch
import late_fusion as lf   # reuse fuse_sample / CAMERAS / MODELS as-is

DATASET   = "trainval"
N_SAMPLES = 4
SCORE_THR = 0.3
OUT_DIR   = os.path.expanduser("~/LCFusion_LT3D/viz")
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = [
    dict(model="centerpoint", dino_run="unbalanced", title="CenterPoint + DINO (unbalanced)"),
    dict(model="bevfusion",   dino_run="oversampled", title="BEVFusion + DINO (oversampled)"),
]

COLORS = {
    "car": "cyan", "truck": "orange", "construction_vehicle": "red",
    "bus": "yellow", "trailer": "magenta", "barrier": "brown",
    "motorcycle": "lime", "bicycle": "pink", "pedestrian": "white",
    "traffic_cone": "gold"
}

NUS_CLASSES = None


def draw_bev_box(ax, box, color, label=None, alpha=1.0):
    x, y, w, l, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[l/2, w/2], [-l/2, w/2], [-l/2, -w/2], [l/2, -w/2]])
    corners = (np.array([[c, -s], [s, c]]) @ corners.T).T + np.array([x, y])
    ax.add_patch(patches.Polygon(corners, closed=True, edgecolor=color,
                                 facecolor='none', linewidth=1.5, alpha=alpha))
    if label:
        ax.text(x, y, label[:3], color=color, fontsize=5, ha='center', va='center')


def draw_panel(ax, points, title, boxes, labels, scores=None):
    ax.set_facecolor('black')
    ax.set_title(title, color='white', fontsize=10)
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
    ax.tick_params(colors='grey')
    m = (np.abs(points[:, 0]) < 50) & (np.abs(points[:, 1]) < 50)
    ax.scatter(points[m, 0], points[m, 1], s=0.1, c='white', alpha=0.3)
    for i in range(len(boxes)):
        cls = NUS_CLASSES[labels[i]] if labels[i] < len(NUS_CLASSES) else str(labels[i])
        a = float(scores[i]) if scores is not None else 1.0
        draw_bev_box(ax, boxes[i], COLORS.get(cls, 'white'), label=cls, alpha=a)
    ax.plot(0, 0, 'r+', markersize=10)


def load_model(model_key):
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')
    cfg_info = lf.MODELS[model_key]
    cfg = patch_cfg(Config.fromfile(cfg_info['cfg']), cfg_info['ckpt'],
                    f"/tmp/vizfused_{model_key}", dataset=DATASET)
    cfg.test_dataloader.batch_size = 1
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(cfg_info['ckpt'])
    runner.model.eval()
    return runner.model, runner.test_dataloader


def get_fused_predictions(model, loader, dino, val_infos, classes, n_samples):
    """Run n_samples through the model, fuse with DINO detections (via
    late_fusion.fuse_sample, unchanged), return list of (points, gt, pred)."""
    results = []
    for idx, data in enumerate(loader):
        if idx >= n_samples:
            break
        with torch.no_grad():
            out = model.test_step(data)[0]

        points = data['inputs']['points'][0].cpu().numpy()
        gt = out.eval_ann_info
        gt_boxes  = gt['gt_bboxes_3d'].tensor.numpy()
        gt_labels = np.array(gt['gt_labels_3d'])
        valid = gt_labels >= 0
        gt_boxes, gt_labels = gt_boxes[valid], gt_labels[valid]

        pred = out.pred_instances_3d
        corners = pred.bboxes_3d.corners.cpu().numpy()
        scores0 = pred.scores_3d.cpu().numpy()
        boxes   = pred.bboxes_3d.tensor.cpu().numpy()
        labels  = pred.labels_3d.cpu().numpy().astype(int)

        sample_idx = out.metainfo['sample_idx']
        info  = val_infos[sample_idx]
        token = info['token']

        if token in dino and len(corners) > 0:
            cam_calib = {
                cam: {"lidar2cam": np.array(info['images'][cam]['lidar2cam']),
                      "cam2img":   np.array(info['images'][cam]['cam2img'])}
                for cam in lf.CAMERAS
            }
            fused_scores = lf.fuse_sample(corners, scores0.copy(), labels, classes,
                                          dino[token], cam_calib)
        else:
            fused_scores = scores0

        m = fused_scores > SCORE_THR
        results.append(dict(points=points, gt_boxes=gt_boxes, gt_labels=gt_labels,
                            boxes=boxes[m], labels=labels[m], scores=fused_scores[m]))
    return results


def main():
    global NUS_CLASSES
    per_config_results = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['title']} ===")
        model, loader = load_model(cfg['model'])
        classes = list(loader.dataset.metainfo['classes'])
        if NUS_CLASSES is None:
            NUS_CLASSES = classes

        dino = pickle.load(open(os.path.expanduser(
            f"~/LCFusion_LT3D/detections/dino_detections_{cfg['dino_run']}.pkl"), "rb"))
        val_infos = pickle.load(open(
            os.path.join(DATA_ROOTS[DATASET], "nuscenes_infos_val.pkl"), "rb"))['data_list']

        res = get_fused_predictions(model, loader, dino, val_infos, classes, N_SAMPLES)
        per_config_results.append(res)
        del model, loader
        torch.cuda.empty_cache()

    print(f"\nRendering {N_SAMPLES} side-by-side comparisons -> {OUT_DIR}\n")
    for idx in range(N_SAMPLES):
        fig, axes = plt.subplots(1, 3, figsize=(22, 8))
        fig.patch.set_facecolor('black')

        r0 = per_config_results[0][idx]   # same GT for both configs (same val order)
        draw_panel(axes[0], r0['points'], "Ground Truth", r0['gt_boxes'], r0['gt_labels'])

        for panel_ax, cfg, res_list in zip(axes[1:], CONFIGS, per_config_results):
            r = res_list[idx]
            draw_panel(panel_ax, r['points'], f"{cfg['title']} (fused, score > {SCORE_THR})",
                      r['boxes'], r['labels'], r['scores'])

        handles = [patches.Patch(edgecolor=c, facecolor='none', label=n) for n, c in COLORS.items()]
        axes[2].legend(handles=handles, loc='upper right', fontsize=6,
                       facecolor='black', labelcolor='white')

        out = os.path.join(OUT_DIR, f"fused_sample_{idx:02d}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='black')
        plt.close()
        print(f"  Saved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
