"""Exact part-1 train composition from the 2D COCO json (image-level)."""
import json, os

ROOT = os.path.expanduser("~/mmdetection3d/data/nuscenes_trainval")
HEAD = {"car", "pedestrian", "bicycle"}
ORDER = ["car","truck","construction_vehicle","bus","trailer","barrier",
         "motorcycle","bicycle","pedestrian","traffic_cone"]

for split in ["train", "val"]:
    coco = json.load(open(os.path.join(ROOT, f"nuscenes_2d_{split}.coco.json")))
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    n_imgs = len(coco["images"])

    # per-image class sets
    img_classes = {im["id"]: set() for im in coco["images"]}
    inst_count = {n: 0 for n in id2name.values()}
    for a in coco["annotations"]:
        nm = id2name[a["category_id"]]
        img_classes[a["image_id"]].add(nm)
        inst_count[nm] += 1

    # per-image composition
    head_only = tail_only = both = empty = 0
    img_freq = {n: 0 for n in id2name.values()}
    for cls_set in img_classes.values():
        has_head = bool(cls_set & HEAD)
        has_tail = bool(cls_set - HEAD) and bool(cls_set)
        for c in cls_set:
            img_freq[c] += 1
        if not cls_set:
            empty += 1
        elif has_head and has_tail:
            both += 1
        elif has_head:
            head_only += 1
        else:
            tail_only += 1

    total_inst = sum(inst_count.values())
    tail_inst = sum(v for k,v in inst_count.items() if k not in HEAD)
    print(f"\n===== {split.upper()} : {n_imgs} images, {total_inst} instances =====")
    print(f"  head-only : {head_only:6d}  ({100*head_only/n_imgs:.1f}%)")
    print(f"  tail-only : {tail_only:6d}  ({100*tail_only/n_imgs:.1f}%)")
    print(f"  both      : {both:6d}  ({100*both/n_imgs:.1f}%)")
    print(f"  empty     : {empty:6d}  ({100*empty/n_imgs:.1f}%)")
    print(f"  tail instances: {tail_inst} ({100*tail_inst/total_inst:.1f}% of all instances)")
    # how many tail instances live in images that also contain a head class?
    print(f"\n  {'class':<22}{'instances':>10}{'img_freq':>10}{'img_freq%':>11}{'repeat@0.17':>12}")
    import math
    for n in ORDER:
        f = img_freq[n]/n_imgs
        r = max(1.0, math.sqrt(0.17/f)) if f>0 else float('nan')
        print(f"  {n:<22}{inst_count[n]:>10}{img_freq[n]:>10}{100*f:>10.1f}%{r:>12.2f}")
