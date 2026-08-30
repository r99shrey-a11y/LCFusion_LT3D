"""
Run DINO on all 6 camera views of every nuScenes val sample.
Saves detections to a per-run pkl for use in late fusion.

DINO_RUN selects the detector:
  "coco"        → COCO-pretrained (80 classes, only 6 nuScenes classes covered)
  "unbalanced"  → fine-tuned on natural (imbalanced) distribution
  "oversampled" → fine-tuned with ClassBalancedDataset (oversample_thr=0.17)

"""

import os, sys, pickle
sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
from utils import REPO, DATA_ROOTS, NUSCENES_CLASSES, torch

DINO_RUN = "oversampled"  # "coco", "unbalanced", "oversampled", "curated", "curated_cbd030"
DATASET  = "trainval"    # "mini" or "trainval"

SCORE_THR = 0.3
CAMERAS   = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# COCO→nuScenes mapping (only used for the COCO-pretrained model)
COCO_TO_NUS = {
    "car": "car", "truck": "truck", "bus": "bus",
    "motorcycle": "motorcycle", "bicycle": "bicycle", "person": "pedestrian",
}

# Config + checkpoint per run
DINO_CFG = {
    "coco": dict(
        cfg=os.path.join(REPO, "checkpoints/dino/dino-4scale_r50_8xb2-12e_coco.py"),
        ckpt=os.path.join(REPO, "checkpoints/dino",
             "dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth"),
        finetuned=False),
    "unbalanced": dict(
        cfg=os.path.expanduser("~/LCFusion_LT3D/configs/dino_nuscenes_unbalanced.py"),
        ckpt=os.path.join(REPO, "work_dirs/dino_nuscenes_unbalanced/epoch_4.pth"),
        finetuned=True),
    "oversampled": dict(
        cfg=os.path.expanduser("~/LCFusion_LT3D/configs/dino_nuscenes_oversampled.py"),
        ckpt=os.path.join(REPO, "work_dirs/dino_nuscenes_oversampled/epoch_4.pth"),
        finetuned=True),
    "curated": dict(
        cfg=os.path.expanduser("~/LCFusion_LT3D/configs/dino_nuscenes_curated.py"),
        ckpt=os.path.join(REPO, "work_dirs/dino_nuscenes_curated/epoch_4.pth"),
        finetuned=True),
    "curated_cbd030": dict(
        cfg=os.path.expanduser("~/LCFusion_LT3D/configs/dino_nuscenes_curated_cbd030.py"),
        ckpt=os.path.join(REPO, "work_dirs/dino_nuscenes_curated_cbd030/epoch_2.pth"),
        finetuned=True),
    "runC_seed1": dict(
        cfg=os.path.expanduser("~/LCFusion_LT3D/configs/dino_nuscenes_oversampled.py"),
        ckpt=os.path.join(REPO, "work_dirs/dino_nuscenes_runC_seed1/epoch_4.pth"),
        finetuned=True),
    "runC_seed2": dict(
        cfg=os.path.expanduser("~/LCFusion_LT3D/configs/dino_nuscenes_oversampled.py"),
        ckpt=os.path.join(REPO, "work_dirs/dino_nuscenes_runC_seed2/epoch_4.pth"),
        finetuned=True),
}


def main():
    from mmdet.apis import init_detector, inference_detector

    info      = DINO_CFG[DINO_RUN]
    data_root = DATA_ROOTS[DATASET]
    device    = "cuda:0" if torch.cuda.is_available() else "cpu"
    model     = init_detector(info['cfg'], info['ckpt'], device=device)
    classes   = model.dataset_meta['classes']

    val_infos = pickle.load(open(os.path.join(data_root, "nuscenes_infos_val.pkl"), "rb"))

    all_detections, class_counts = {}, {}

    for vinfo in val_infos['data_list']:
        token = vinfo['token']
        dets  = []
        for cam in CAMERAS:
            img_path = os.path.join(data_root, "samples", cam,
                                    os.path.basename(vinfo['images'][cam]['img_path']))
            result = inference_detector(model, img_path)
            pred   = result.pred_instances
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            labels = pred.labels.cpu().numpy().astype(int)

            mask = scores > SCORE_THR
            for b, s, l in zip(bboxes[mask], scores[mask], labels[mask]):
                cls_name = classes[l]
                if info['finetuned']:
                    nus_cls = cls_name          # already a nuScenes class
                else:
                    nus_cls = COCO_TO_NUS.get(cls_name)
                dets.append(dict(cam=cam, bbox=b, score=float(s),
                                 coco_class=cls_name, nus_class=nus_cls))
                if nus_cls:
                    class_counts[nus_cls] = class_counts.get(nus_cls, 0) + 1
        all_detections[token] = dets

    out_path = os.path.expanduser(f"~/LCFusion_LT3D/detections/dino_detections_{DINO_RUN}.pkl")
    pickle.dump(all_detections, open(out_path, "wb"))

    print(f"\nDINO ({DINO_RUN}) on {DATASET} val — 6 cameras, {len(all_detections)} samples")
    print(f"Saved to: {out_path}\n")
    print("Detection counts per nuScenes class:")
    for cls in NUSCENES_CLASSES:
        print(f"  {cls:<22} {class_counts.get(cls, 0)}")


if __name__ == "__main__":
    main()
