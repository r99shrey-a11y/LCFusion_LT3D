"""
Late fusion of a 3D detector (BEVFusion or CenterPoint) with DINO 2D detections.
Adapts the methodology of Ma et al. (lt3d-lf) to the chosen 3D backbone.

Pipeline (per sample):
  1. Project each 3D box onto the 6 camera images (lidar2cam + cam2img).
  2. Greedily match projected 3D boxes to DINO 2D boxes by IoU (same class).
  3. Fuse matched scores with a Bayesian combination (per-class c, prior p).
  4. Down-weight unmatched-but-visible 3D boxes; leave DINO-uncovered classes as-is.
  5. Evaluate the fused detections with the nuScenes metric.
"""

import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.expanduser("~/LCFusion_LT3D"))
from utils import REPO, DATA_ROOTS, patch_cfg, print_results, torch

# CONFIG
MODEL    = "bevfusion"   # 3D backbone: "bevfusion" or "centerpoint"
DATASET  = "trainval"    # "mini" or "trainval"
DINO_RUN = "curated_p1_cbd"  # "coco", "runA", "runB", "runC", "curated", "curated_p1_cbd"

IOU_THRESH     = 0.5     # min IoU for a 3D↔2D match
SCORE_3D_THR   = 0.05    # ignore very low-score 3D boxes before fusion
W_UNMATCHED    = 0.5     # score multiplier for visible-but-unmatched 3D boxes
IMG_W, IMG_H   = 1600, 900   # nuScenes camera resolution
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# Classes DINO covers. COCO-pretrained covers only 6; fine-tuned covers all 10.
DINO_COVERED_COCO = {"car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"}
DINO_COVERED_ALL  = {"car", "truck", "trailer", "bus", "construction_vehicle",
                     "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"}
DINO_COVERED = DINO_COVERED_COCO if DINO_RUN == "coco" else DINO_COVERED_ALL

# Per-class Bayesian calibration (c = 2D score weight, p = class prior).
# Values adapted from lt3d-lf cal.py; can be tuned on val.
CAL = {
    "car":                  {"c": 0.6, "p": 0.1},
    "truck":                {"c": 1.2, "p": 0.1},
    "trailer":              {"c": 0.6, "p": 0.1},
    "bus":                  {"c": 0.5, "p": 0.1},
    "construction_vehicle": {"c": 1.0, "p": 0.1},
    "bicycle":              {"c": 0.8, "p": 0.1},
    "motorcycle":           {"c": 0.7, "p": 0.1},
    "pedestrian":           {"c": 0.3, "p": 0.1},
    "traffic_cone":         {"c": 1.1, "p": 0.1},
    "barrier":              {"c": 1.1, "p": 0.1},
}

MODELS = {
    "bevfusion": dict(
        cfg=os.path.join(REPO, "projects/BEVFusion/configs",
            "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"),
        ckpt=os.path.join(REPO, "checkpoints/bevfusion",
            "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth"),
        title="BEVFusion + DINO"),
    "centerpoint": dict(
        cfg=os.path.join(REPO, "checkpoints/centerpoint",
            "centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"),
        ckpt=os.path.join(REPO, "checkpoints/centerpoint",
            "centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth"),
        title="CenterPoint + DINO"),
}


# Geometry helpers (ported from lt3d-lf make_fine_fusion_res_done.py)
# sensor2lidar_rotation/translation + cam_intrinsic from its info dict,
# while mmdet3d's pkl stores a combined lidar2cam (4x4) and cam2img (3x3)
# matrix directly - so we invert lidar2cam to get camera2lidar to keep the
# rest of the function body (homogeneous transform, perspective divide)
# unchanged from the original.

def lidar2img(points_lidar, camera_info):
    """Project LiDAR-frame points into the image plane.

    Ported directly from lt3d-lf's lidar2img(), same steps in the same
    order: build homogeneous points, invert camera2lidar to get
    lidar2camera, transform, filter points behind the camera, perspective
    divide, apply the intrinsic matrix.
    """
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)

    # lt3d-lf builds camera2lidar from separate rotation/translation; our
    # data already provides lidar2camera (lidar2cam) directly.
    lidar2camera = camera_info['lidar2cam']

    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]

    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = camera_info['cam_intrinsic']
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


def check_point_in_img(points, height, width):
    """Directly from lt3d-lf's check_point_in_img() (unchanged)."""
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(valid,
                           np.logical_and(points[:, 0] < width,
                                          points[:, 1] < height))
    return valid


def compute_iou(rec1, rec2):
    """Directly from lt3d-lf's compute_iou() (unchanged).

    rec1, rec2: (y0, x0, y1, x1) i.e. (top, left, bottom, right).
    """
    rec1 = (rec1[1], rec1[0], rec1[3], rec1[2])
    rec2 = (rec2[1], rec2[0], rec2[3], rec2[2])

    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    sum_area = S_rec1 + S_rec2

    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
    return (intersect / (sum_area - intersect)) * 1.0


def project_box(corners_lidar, lidar2cam, cam2img):
    """Project one 3D box's 8 corners to a 2D [x1,y1,x2,y2] image bbox,
    using lidar2img()/check_point_in_img() above (our wrapper, not present
    in lt3d-lf, needed because they operate on the whole-scene corner array
    while we call this per-box)."""
    camera_info = {'lidar2cam': lidar2cam, 'cam_intrinsic': cam2img}
    points_img, valid_z = lidar2img(corners_lidar, camera_info)
    if valid_z.sum() < 1:
        return None
    pts = points_img[valid_z]
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    x1, x2 = max(0, min(x1, IMG_W)), max(0, min(x2, IMG_W))
    y1, y2 = max(0, min(y1, IMG_H)), max(0, min(y2, IMG_H))
    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return None
    return [x1, y1, x2, y2]


def iou(a, b):
    """Wrapper for compute_iou() taking [x1,y1,x2,y2] boxes (our convention)
    and converting to the (y0,x0,y1,x1) convention compute_iou() expects."""
    return compute_iou((a[1], a[0], a[3], a[2]), (b[1], b[0], b[3], b[2]))


def bayes_fuse(s3d, s2d, cls):
    """Bayesian score fusion for a matched 3D/2D pair of the same class."""
    c, p = CAL[cls]["c"], CAL[cls]["p"]
    f2d, f3d = c * s2d, s3d
    num = f2d * f3d / p
    den = num + (1 - f2d) * (1 - f3d) / (1 - p)
    return float(num / den) if den > 0 else s3d


# Fusion for one sample
def fuse_sample(boxes_corners, scores, labels, classes, dino_dets, cam_calib):
    """Return updated scores after fusing with DINO detections.

    Matching follows lt3d-lf (Ma et al.): per camera view, build an IoU cost
    matrix between all visible 3D-projected boxes and all 2D DINO boxes of
    the same class, then solve optimal one-to-one assignment (Hungarian
    algorithm via scipy.optimize.linear_sum_assignment, equivalent to their
    lap.lapjv), keeping only matches with IoU > IOU_THRESH.

    boxes_corners : (N,8,3) 3D box corners in LiDAR frame
    scores, labels: (N,) arrays
    classes       : list mapping label index → class name
    dino_dets     : list of {cam, bbox, score, nus_class}
    cam_calib     : dict cam → {lidar2cam (4,4), cam2img (3,3)}
    """
    from scipy.optimize import linear_sum_assignment

    dino_by_cam = {c: [] for c in CAMERAS}
    for d in dino_dets:
        if d["nus_class"] in DINO_COVERED:
            dino_by_cam[d["cam"]].append(d)

    new_scores = scores.copy()
    matched_flag = np.zeros(len(scores), dtype=bool)
    visible_flag = np.zeros(len(scores), dtype=bool)

    valid_idx = [i for i in range(len(scores))
                if classes[labels[i]] in DINO_COVERED and scores[i] >= SCORE_3D_THR]

    for cam in CAMERAS:
        dino_2d = dino_by_cam[cam]
        if not dino_2d:
            continue

        # project all eligible 3D boxes into this camera
        proj = {}   # i -> box2d
        for i in valid_idx:
            box2d = project_box(boxes_corners[i], cam_calib[cam]["lidar2cam"], cam_calib[cam]["cam2img"])
            if box2d is not None:
                proj[i] = box2d
                visible_flag[i] = True
        if not proj:
            continue

        # Match separately within each class (same-class constraint, as in lt3d-lf)
        by_class = {}
        for i, box2d in proj.items():
            by_class.setdefault(classes[labels[i]], []).append((i, box2d))

        for cls, entries in by_class.items():
            d2d = [d for d in dino_2d if d["nus_class"] == cls]
            if not d2d:
                continue
            idxs_3d = [e[0] for e in entries]
            boxes_3d_2d = [e[1] for e in entries]

            # cost matrix = 1 - IoU (Hungarian minimizes cost)
            n, m = len(boxes_3d_2d), len(d2d)
            cost = np.ones((n, m))
            for a in range(n):
                for b in range(m):
                    cost[a, b] = 1.0 - iou(boxes_3d_2d[a], d2d[b]["bbox"])

            row_idx, col_idx = linear_sum_assignment(cost)
            for a, b in zip(row_idx, col_idx):
                if cost[a, b] > (1.0 - IOU_THRESH):
                    continue   # below IoU threshold, reject the match
                i = idxs_3d[a]
                fused = bayes_fuse(scores[i], d2d[b]["score"], cls)
                if fused > new_scores[i] or not matched_flag[i]:
                    new_scores[i] = fused
                matched_flag[i] = True

    for i in valid_idx:
        if visible_flag[i] and not matched_flag[i]:
            new_scores[i] = scores[i] * W_UNMATCHED   # visible but no 2D support

    return new_scores


# Main
def main():
    cfg_info = MODELS[MODEL]
    sys.path.insert(0, REPO)
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.registry import init_default_scope, METRICS

    init_default_scope('mmdet3d')
    cfg = patch_cfg(Config.fromfile(cfg_info['cfg']), cfg_info['ckpt'],
                    f"/tmp/fuse_{MODEL}", dataset=DATASET)
    cfg.test_dataloader.batch_size = 1

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(cfg_info['ckpt'])
    model  = runner.model
    model.eval()
    loader = runner.test_dataloader
    classes = list(loader.dataset.metainfo['classes'])

    # DINO detections keyed by sample token
    dino = pickle.load(open(os.path.expanduser(
        f"~/LCFusion_LT3D/detections/dino_detections_{DINO_RUN}.pkl"), "rb"))

    # Val info for calibration + token lookup (indexed by sample_idx)
    val_infos = pickle.load(open(
        os.path.join(DATA_ROOTS[DATASET], "nuscenes_infos_val.pkl"), "rb"))['data_list']

    # Build evaluator
    evaluator = METRICS.build(cfg.test_evaluator)
    evaluator.dataset_meta = loader.dataset.metainfo

    print(f"\nRunning {cfg_info['title']} late fusion on {DATASET} val "
          f"({len(val_infos)} samples)...\n")

    for data in loader:
        with torch.no_grad():
            out = model.test_step(data)[0]

        idx   = out.metainfo['sample_idx']
        info  = val_infos[idx]
        token = info['token']

        pred    = out.pred_instances_3d
        corners = pred.bboxes_3d.corners.cpu().numpy()      # (N,8,3)
        scores  = pred.scores_3d.cpu().numpy()
        labels  = pred.labels_3d.cpu().numpy().astype(int)

        if token in dino and len(corners) > 0:
            cam_calib = {
                cam: {
                    "lidar2cam": np.array(info['images'][cam]['lidar2cam']),
                    "cam2img":   np.array(info['images'][cam]['cam2img']),
                } for cam in CAMERAS
            }
            new_scores = fuse_sample(corners, scores, labels, classes,
                                     dino[token], cam_calib)
            pred.scores_3d = torch.from_numpy(new_scores).to(pred.scores_3d.device)
            out.pred_instances_3d = pred

        evaluator.process(data_samples=[out.to_dict()], data_batch=data)

    metrics = evaluator.evaluate(len(val_infos))
    print_results(cfg_info['title'], metrics, dataset=DATASET)


if __name__ == "__main__":
    main()
