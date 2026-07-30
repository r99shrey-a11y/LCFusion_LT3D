"""Quick fusion-calibration grid search for BEVFusion + DINO(curated).
Runs BEVFusion inference ONCE (caches 3D preds in memory), then re-fuses +
re-evaluates for several CAL / W_UNMATCHED settings. No retraining/re-inference.
Focus: recover the barrier regression (0.590 -> 0.423)."""
import os, sys, copy, pickle
import numpy as np
sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
import late_fusion as lf
from utils import REPO, DATA_ROOTS, patch_cfg, torch, NUSCENES_CLASSES, HEAD_CLASSES

DATASET = "trainval"
PREFIX  = "NuScenes metric/pred_instances_3d_NuScenes/"
DISTS   = ["AP_dist_0.5","AP_dist_1.0","AP_dist_2.0","AP_dist_4.0"]

def setup():
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.registry import init_default_scope, METRICS
    init_default_scope('mmdet3d')
    ci = lf.MODELS["bevfusion"]
    cfg = patch_cfg(Config.fromfile(ci['cfg']), ci['ckpt'], "/tmp/grid_bev", dataset=DATASET)
    cfg.test_dataloader.batch_size = 1
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(ci['ckpt']); runner.model.eval()
    return cfg, runner

def main():
    cfg, runner = setup()
    model, loader = runner.model, runner.test_dataloader
    classes = list(loader.dataset.metainfo['classes'])
    from mmengine.registry import METRICS
    dino = pickle.load(open(os.path.expanduser("~/LCFusion_LT3D/detections/dino_detections_curated.pkl"),"rb"))
    val_infos = pickle.load(open(os.path.join(DATA_ROOTS[DATASET],"nuscenes_infos_val.pkl"),"rb"))['data_list']

    # ---- Phase 1: BEVFusion inference once, cache ----
    print("Phase 1: BEVFusion inference (once)...", flush=True)
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
            for k,v in cal_over.items(): lf.CAL[k].update(v)
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
        print(f"[{name:<28}] barrier={ap['barrier']:.3f} truck={ap['truck']:.3f} "
              f"cone={ap['traffic_cone']:.3f} | head={head:.3f} tail={tail:.3f} "
              f"overall={m[f'{PREFIX}mAP']:.3f} NDS={m[f'{PREFIX}NDS']:.3f}", flush=True)

    print("Phase 2: calibration sweep\n")
    run_cfg("baseline (barrier c=1.1,W=0.5)")
    run_cfg("barrier c=0.7",  {"barrier":{"c":0.7}})
    run_cfg("barrier c=0.5",  {"barrier":{"c":0.5}})
    run_cfg("barrier c=0.3",  {"barrier":{"c":0.3}})
    run_cfg("W_unmatched=1.0 (no penalty)", None, 1.0)
    run_cfg("W_unmatched=0.8", None, 0.8)
    run_cfg("barrier c=0.5 + W=0.8", {"barrier":{"c":0.5}}, 0.8)

if __name__ == "__main__":
    main()
