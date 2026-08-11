"""Quick fusion-calibration grid search for BEVFusion + DINO(curated_cbd030).
Runs BEVFusion inference ONCE (caches 3D preds in memory), then re-fuses +
re-evaluates for several barrier CAL settings. No retraining/re-inference.

Diagnosis (analysis/diagnose_fusion.py + hand calc, see thesis_context.md):
barrier's matched-2D-score average dropped RunC 0.634 -> cbd030 0.454 even
though match rate roughly doubled and matched IoU improved. Since
bayes_fuse is sensitive to c*s2d, this sweep tests whether RAISING c
(not lowering, as the earlier "curated" grid search tried) compensates for
the lower average s2d and recovers barrier's fused AP.

Target: c*s2d(cbd030) ~= c*s2d(RunC) => c ~= 1.1 * 0.634/0.454 ~= 1.54
"""
import os, sys, copy, pickle
import numpy as np
sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
import late_fusion as lf
from utils import REPO, DATA_ROOTS, patch_cfg, torch, NUSCENES_CLASSES, HEAD_CLASSES

DATASET  = "trainval"
DINO_RUN = "curated_cbd030"
PREFIX   = "NuScenes metric/pred_instances_3d_NuScenes/"
DISTS    = ["AP_dist_0.5", "AP_dist_1.0", "AP_dist_2.0", "AP_dist_4.0"]


def setup():
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')
    ci = lf.MODELS["bevfusion"]
    cfg = patch_cfg(Config.fromfile(ci['cfg']), ci['ckpt'], "/tmp/grid_bev2", dataset=DATASET)
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

    def run_cfg(name, cal_over=None, w_unmatched=0.5):
        lf.CAL = copy.deepcopy(ORIG_CAL)
        if cal_over:
            for k, v in cal_over.items():
                lf.CAL[k].update(v)
        lf.W_UNMATCHED = w_unmatched
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
        print(f"[{name:<32}] barrier={ap['barrier']:.3f} truck={ap['truck']:.3f} "
              f"cv={ap['construction_vehicle']:.3f} moto={ap['motorcycle']:.3f} "
              f"cone={ap['traffic_cone']:.3f} | head={head:.3f} tail={tail:.3f} "
              f"overall={m[f'{PREFIX}mAP']:.3f} NDS={m[f'{PREFIX}NDS']:.3f}", flush=True)

    print("Phase 2: barrier calibration sweep (RAISING c, not lowering)\n")
    run_cfg("baseline (barrier c=1.1, W=0.5)")
    run_cfg("barrier c=1.3",              {"barrier": {"c": 1.3}})
    run_cfg("barrier c=1.5",              {"barrier": {"c": 1.5}})
    run_cfg("barrier c=1.54 (matched c*s2d)", {"barrier": {"c": 1.54}})
    run_cfg("barrier c=1.7",              {"barrier": {"c": 1.7}})
    run_cfg("barrier c=2.0",              {"barrier": {"c": 2.0}})
    run_cfg("barrier c=1.5 + W=1.0",      {"barrier": {"c": 1.5}}, 1.0)
    run_cfg("W_unmatched=1.0 only",       None, 1.0)


if __name__ == "__main__":
    main()
