"""Final quick fusion-hyperparameter grid search for BEVFusion + DINO(Run C).
Runs BEVFusion inference ONCE (caches 3D preds in memory), then re-fuses +
re-evaluates for several IOU_THRESH / SCORE_3D_THR / W_UNMATCHED settings.
No retraining/re-inference. These 3 constants were fixed at project start
and never swept for Run C specifically (only CAL/W_unmatched were swept, and
only on the curated variants while diagnosing barrier)."""
import os, sys, copy, pickle
import numpy as np
sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
import late_fusion as lf
from utils import REPO, DATA_ROOTS, patch_cfg, torch, NUSCENES_CLASSES, HEAD_CLASSES

DATASET  = "trainval"
DINO_RUN = "oversampled"
PREFIX   = "NuScenes metric/pred_instances_3d_NuScenes/"
DISTS    = ["AP_dist_0.5", "AP_dist_1.0", "AP_dist_2.0", "AP_dist_4.0"]


def setup():
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')
    ci = lf.MODELS["bevfusion"]
    cfg = patch_cfg(Config.fromfile(ci['cfg']), ci['ckpt'], "/tmp/grid_runc_final", dataset=DATASET)
    cfg.test_dataloader.batch_size = 1
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(ci['ckpt']); runner.model.eval()
    return cfg, runner


def main():
    cfg, runner = setup()
    model, loader = runner.model, runner.test_dataloader
    classes = list(loader.dataset.metainfo['classes'])
    from mmengine.registry import METRICS
    dino = pickle.load(open(os.path.expanduser(
        f"~/LCFusion_LT3D/detections/dino_detections_{DINO_RUN}.pkl"), "rb"))
    val_infos = pickle.load(open(
        os.path.join(DATA_ROOTS[DATASET], "nuscenes_infos_val.pkl"), "rb"))['data_list']

    print(f"Phase 1: BEVFusion inference (once), DINO_RUN={DINO_RUN}...", flush=True)
    cache = []
    for data in loader:
        with torch.no_grad():
            out = model.test_step(data)[0]
        idx = out.metainfo['sample_idx']; token = val_infos[idx]['token']
        pred = out.pred_instances_3d
        corners = pred.bboxes_3d.corners.cpu().numpy()
        scores0 = pred.scores_3d.cpu().numpy()
        labels  = pred.labels_3d.cpu().numpy().astype(int)
        info = val_infos[idx]
        cam_calib = {cam: {"lidar2cam": np.array(info['images'][cam]['lidar2cam']),
                           "cam2img": np.array(info['images'][cam]['cam2img'])}
                     for cam in lf.CAMERAS}
        out.pred_instances_3d = pred.to('cpu')
        cache.append((out, token, corners, scores0, labels, cam_calib, data))
    print(f"cached {len(cache)} samples\n", flush=True)

    ORIG_CAL = copy.deepcopy(lf.CAL)
    ORIG_IOU = lf.IOU_THRESH
    ORIG_SCORE3D = lf.SCORE_3D_THR
    ORIG_WUN = lf.W_UNMATCHED

    def run_cfg(name, iou_thresh=None, score3d=None, w_unmatched=None):
        lf.CAL = copy.deepcopy(ORIG_CAL)
        lf.IOU_THRESH = iou_thresh if iou_thresh is not None else ORIG_IOU
        lf.SCORE_3D_THR = score3d if score3d is not None else ORIG_SCORE3D
        lf.W_UNMATCHED = w_unmatched if w_unmatched is not None else ORIG_WUN
        evaluator = METRICS.build(cfg.test_evaluator)
        evaluator.dataset_meta = loader.dataset.metainfo
        for out, token, corners, scores0, labels, cam_calib, data in cache:
            if token in dino and len(corners) > 0:
                new = lf.fuse_sample(corners, scores0.copy(), labels, classes, dino[token], cam_calib)
                out.pred_instances_3d.scores_3d = torch.from_numpy(new)
            else:
                out.pred_instances_3d.scores_3d = torch.from_numpy(scores0.copy())
            evaluator.process(data_samples=[out.to_dict()], data_batch=data)
        m = evaluator.evaluate(len(cache))
        ap = {c: float(np.mean([m[f"{PREFIX}{c}_{d}"] for d in DISTS])) for c in NUSCENES_CLASSES}
        head = np.mean([ap[c] for c in NUSCENES_CLASSES if c in HEAD_CLASSES])
        tail = np.mean([ap[c] for c in NUSCENES_CLASSES if c not in HEAD_CLASSES])
        print(f"[{name:<32}] head={head:.3f} tail={tail:.3f} "
              f"overall={m[f'{PREFIX}mAP']:.3f} NDS={m[f'{PREFIX}NDS']:.3f}", flush=True)

    print(f"Phase 2: fusion hyperparameter sweep on Run C "
          f"(baseline IOU={ORIG_IOU}, SCORE_3D={ORIG_SCORE3D}, W_unmatched={ORIG_WUN})\n")
    run_cfg("baseline (IOU=0.5, S3D=0.05, W=0.5)")

    # IOU_THRESH sweep
    run_cfg("IOU_THRESH=0.3", iou_thresh=0.3)
    run_cfg("IOU_THRESH=0.4", iou_thresh=0.4)
    run_cfg("IOU_THRESH=0.6", iou_thresh=0.6)
    run_cfg("IOU_THRESH=0.7", iou_thresh=0.7)

    # SCORE_3D_THR sweep
    run_cfg("SCORE_3D_THR=0.02", score3d=0.02)
    run_cfg("SCORE_3D_THR=0.10", score3d=0.10)
    run_cfg("SCORE_3D_THR=0.15", score3d=0.15)

    # W_UNMATCHED sweep
    run_cfg("W_unmatched=0.3", w_unmatched=0.3)
    run_cfg("W_unmatched=0.7", w_unmatched=0.7)
    run_cfg("W_unmatched=1.0 (no penalty)", w_unmatched=1.0)

    # best-looking combinations (filled after seeing individual sweeps would be ideal,
    # but run a couple of plausible joint combos now to save a second pass)
    run_cfg("IOU=0.4 + W=0.7", iou_thresh=0.4, w_unmatched=0.7)
    run_cfg("IOU=0.6 + S3D=0.10", iou_thresh=0.6, score3d=0.10)


if __name__ == "__main__":
    main()
