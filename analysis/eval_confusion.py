"""
Per-class TP/FP/FN, precision, recall, and a cross-class confusion matrix for
the FUSED 3D detections (BEVFusion + DINO), at a fixed score threshold.

Matching protocol replicates nuScenes' own AP matching (nuscenes.eval.detection.algo.accumulate):
  - greedy, confidence-sorted: highest-score prediction claims the CLOSEST
    unclaimed GT box of the SAME class within dist_th (center distance, xy).
  - dist_th = 2.0m (the nuScenes dist_th_tp convention; also one of the 4
    thresholds {0.5,1,2,4} averaged into the AP numbers reported throughout
    this project).
  - unlike AP (which sweeps every confidence threshold), TP/FP/FN/precision/
    recall need ONE fixed score threshold: SCORE_THR=0.3, matching the
    convention already used in visualize.py / visualize_fused.py.

Confusion matrix: for every prediction that is a class-restricted FALSE
POSITIVE (no same-class GT within dist_th), find its nearest GT box of ANY
class within dist_th — if one exists, count it as a (predicted_class,
true_class) confusion; otherwise count it as a "background" false positive
(no GT object nearby at all, i.e. a hallucinated detection). This mirrors
what a real per-detection error analysis looks like, since nuScenes' own
matching never reports cross-class confusions itself (it only ever compares
same-class boxes).

Usage: /home/batashey/miniconda3/envs/lcfusion/bin/python analysis/eval_confusion.py <DINO_RUN>
"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
import late_fusion as lf
from utils import REPO, DATA_ROOTS, patch_cfg, torch, NUSCENES_CLASSES

DATASET    = "trainval"
DIST_TH    = 2.0     # meters, nuScenes dist_th_tp convention
SCORE_THR  = 0.3      # fixed confidence threshold for this snapshot


def setup():
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')
    ci = lf.MODELS[os.environ.get("CONFMAT_MODEL", "bevfusion")]
    cfg = patch_cfg(Config.fromfile(ci['cfg']), ci['ckpt'], "/tmp/confmat", dataset=DATASET)
    cfg.test_dataloader.batch_size = 1
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(ci['ckpt']); runner.model.eval()
    return runner


def main(dino_run):
    runner = setup()
    model, loader = runner.model, runner.test_dataloader
    classes = list(loader.dataset.metainfo['classes'])

    dino = pickle.load(open(os.path.expanduser(
        f"~/LCFusion_LT3D/detections/dino_detections_{dino_run}.pkl"), "rb"))
    val_infos = pickle.load(open(
        os.path.join(DATA_ROOTS[DATASET], "nuscenes_infos_val.pkl"), "rb"))['data_list']

    # accumulators
    tp = {c: 0 for c in NUSCENES_CLASSES}
    fp_class = {c: 0 for c in NUSCENES_CLASSES}   # unmatched predictions (any reason)
    fn = {c: 0 for c in NUSCENES_CLASSES}
    n_gt = {c: 0 for c in NUSCENES_CLASSES}
    confusion = {p: {g: 0 for g in NUSCENES_CLASSES} for p in NUSCENES_CLASSES}  # [pred][true]
    background_fp = {c: 0 for c in NUSCENES_CLASSES}  # predicted class -> no GT nearby at all

    print(f"Running BEVFusion + DINO({dino_run}) fusion and matching "
          f"(dist_th={DIST_TH}m, score_thr={SCORE_THR})...\n", flush=True)

    for data in loader:
        with torch.no_grad():
            out = model.test_step(data)[0]

        idx = out.metainfo['sample_idx']
        info = val_infos[idx]
        token = info['token']

        pred = out.pred_instances_3d
        corners = pred.bboxes_3d.corners.cpu().numpy()
        scores0 = pred.scores_3d.cpu().numpy()
        labels = pred.labels_3d.cpu().numpy().astype(int)
        boxes = pred.bboxes_3d.tensor.cpu().numpy()  # x,y,z,w,l,h,yaw,...

        if token in dino and len(corners) > 0:
            cam_calib = {cam: {"lidar2cam": np.array(info['images'][cam]['lidar2cam']),
                               "cam2img": np.array(info['images'][cam]['cam2img'])}
                        for cam in lf.CAMERAS}
            scores = lf.fuse_sample(corners, scores0.copy(), labels, classes, dino[token], cam_calib)
        else:
            scores = scores0

        keep = scores > SCORE_THR
        p_boxes = boxes[keep][:, :2]     # xy centers
        p_labels = labels[keep]
        p_scores = scores[keep]

        # GT for this sample
        gt = out.eval_ann_info
        gt_boxes_full = gt['gt_bboxes_3d'].tensor.numpy()
        gt_labels = np.array(gt['gt_labels_3d'])
        valid = gt_labels >= 0
        gt_xy = gt_boxes_full[valid][:, :2]
        gt_lab = gt_labels[valid]

        for c in NUSCENES_CLASSES:
            ci = classes.index(c)
            n_gt[c] += int((gt_lab == ci).sum())

        # greedy, confidence-sorted, per-class matching (mirrors nuscenes algo.accumulate)
        order = np.argsort(-p_scores)
        taken_gt = set()
        matched_pred = np.zeros(len(p_labels), dtype=bool)

        for pi in order:
            cls_i = p_labels[pi]
            best_d, best_g = np.inf, None
            for gi in range(len(gt_lab)):
                if gt_lab[gi] != cls_i or gi in taken_gt:
                    continue
                d = np.linalg.norm(p_boxes[pi] - gt_xy[gi])
                if d < best_d:
                    best_d, best_g = d, gi
            if best_g is not None and best_d < DIST_TH:
                taken_gt.add(best_g)
                matched_pred[pi] = True
                tp[classes[cls_i]] += 1

        # unmatched predictions -> false positives; find nearest GT of ANY class for confusion
        for pi in range(len(p_labels)):
            if matched_pred[pi]:
                continue
            cls_i = p_labels[pi]
            pred_name = classes[cls_i]
            fp_class[pred_name] += 1
            best_d, best_gt_lab = np.inf, None
            for gi in range(len(gt_lab)):
                d = np.linalg.norm(p_boxes[pi] - gt_xy[gi])
                if d < best_d:
                    best_d, best_gt_lab = d, gt_lab[gi]
            if best_gt_lab is not None and best_d < DIST_TH:
                true_name = classes[best_gt_lab]
                confusion[pred_name][true_name] += 1
            else:
                background_fp[pred_name] += 1

        # unmatched GT -> false negatives
        for gi in range(len(gt_lab)):
            if gi not in taken_gt:
                fn[classes[gt_lab[gi]]] += 1

    # ---------------- report ----------------
    print(f"{'class':<22}{'GT':>7}{'TP':>7}{'FP':>7}{'FN':>7}{'Precision':>11}{'Recall':>9}")
    for c in NUSCENES_CLASSES:
        p = tp[c] / (tp[c] + fp_class[c]) if (tp[c] + fp_class[c]) > 0 else float('nan')
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else float('nan')
        print(f"{c:<22}{n_gt[c]:>7}{tp[c]:>7}{fp_class[c]:>7}{fn[c]:>7}{p:>11.3f}{r:>9.3f}")

    tot_tp = sum(tp.values()); tot_fp = sum(fp_class.values()); tot_fn = sum(fn.values())
    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else float('nan')
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else float('nan')
    print(f"\nMicro-avg precision: {micro_p:.3f}   Micro-avg recall: {micro_r:.3f}")
    print(f"Total: TP={tot_tp} FP={tot_fp} FN={tot_fn}")

    print(f"\n=== Confusion matrix (rows=predicted, cols=true class of nearest GT within {DIST_TH}m) ===")
    header_label = "pred/true"
    print(f"{header_label:<14}" + "".join(f"{c[:8]:>9}" for c in NUSCENES_CLASSES) + f"{'bg(none)':>10}")
    for p_c in NUSCENES_CLASSES:
        row = "".join(f"{confusion[p_c][t_c]:>9}" for t_c in NUSCENES_CLASSES)
        print(f"{p_c[:13]:<14}{row}{background_fp[p_c]:>10}")

    return dict(tp=tp, fp=fp_class, fn=fn, n_gt=n_gt, confusion=confusion, background_fp=background_fp)


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "oversampled"
    main(run)
