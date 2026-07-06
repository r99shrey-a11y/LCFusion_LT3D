"""
Run DINO on all 6 camera views of every nuScenes val sample.
Saves detections to dino_detections.pkl for use in late fusion.
Prints per-class detection counts as a sanity check.

Usage: /home/batashey/miniconda3/envs/lcfusion/bin/python eval_dino.py
"""

import os, sys, pickle
sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
from utils import REPO, DATA_ROOTS, torch

# ── CHOOSE DATASET HERE ───────────────────────────────────────────────────────
DATASET = "trainval"     # options: "mini" or "trainval"
# ──────────────────────────────────────────────────────────────────────────────

CKPT_DIR  = os.path.join(REPO, "checkpoints/dino")
SCORE_THR = 0.3
CAMERAS   = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

COCO_TO_NUS = {
    "car":        "car",
    "truck":      "truck",
    "bus":        "bus",
    "motorcycle": "motorcycle",
    "bicycle":    "bicycle",
    "person":     "pedestrian",
}


def main():
    from mmdet.apis import init_detector, inference_detector

    data_root = DATA_ROOTS[DATASET]
    cfg_path  = os.path.join(CKPT_DIR, "dino-4scale_r50_8xb2-12e_coco.py")
    ckpt_path = os.path.join(CKPT_DIR,
                next(f for f in os.listdir(CKPT_DIR) if f.endswith(".pth")))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model     = init_detector(cfg_path, ckpt_path, device=device)
    val_infos = pickle.load(open(os.path.join(data_root, "nuscenes_infos_val.pkl"), "rb"))

    all_detections = {}   # token -> list of dicts {cam, bbox, score, coco_class, nus_class}
    class_counts   = {}

    for info in val_infos['data_list']:
        token = info['token']
        dets  = []

        for cam in CAMERAS:
            img_path = os.path.join(data_root, "samples", cam,
                                    os.path.basename(info['images'][cam]['img_path']))

            result = inference_detector(model, img_path)
            pred   = result.pred_instances
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            labels = pred.labels.cpu().numpy().astype(int)

            mask = scores > SCORE_THR
            for b, s, l in zip(bboxes[mask], scores[mask], labels[mask]):
                coco_cls = model.dataset_meta['classes'][l]
                nus_cls  = COCO_TO_NUS.get(coco_cls)
                dets.append(dict(cam=cam, bbox=b, score=float(s),
                                 coco_class=coco_cls, nus_class=nus_cls))
                class_counts[coco_cls] = class_counts.get(coco_cls, 0) + 1

        all_detections[token] = dets

    out_path = os.path.join(os.path.expanduser("~/LCFusion_LT3D"), "dino_detections.pkl")
    pickle.dump(all_detections, open(out_path, "wb"))

    print(f"\nDINO on nuScenes {DATASET} val — all 6 cameras  ({len(all_detections)} samples)")
    print(f"Score threshold: {SCORE_THR}\nSaved to: {out_path}\n")
    print("Detection counts per COCO class:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        nus = COCO_TO_NUS.get(cls, "—")
        print(f"  {cls:<20} → nuScenes: {str(nus):<15}  count: {count}")


if __name__ == "__main__":
    main()
