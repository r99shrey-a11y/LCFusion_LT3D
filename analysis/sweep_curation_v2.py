"""Simulate improved curation policies on the full pool (parts 1,2,3,5,6):
(A) lower the car/ped-only+empty trim ratio (0.25 -> lower values)
(B) additionally cap "both" images by head:tail box-count ratio (drop images
    where head boxes swamp tail boxes beyond a threshold, e.g. car_count >
    K * tail_count) — the "25 cars, 1 tail box" case.
Reports per-class instances, image-freq, and the freq/car ratio used earlier."""
import json, os, math, random
from collections import defaultdict

ROOT = os.path.expanduser("~/mmdetection3d/data/nuscenes_trainval")
POOL = os.path.join(ROOT, "nuscenes_2d_train_pool.coco.json")
HEAD = {"car", "pedestrian", "bicycle"}
PRESERVE = {"bicycle"}
ORDER = ["car","pedestrian","barrier","traffic_cone","truck","trailer",
         "construction_vehicle","bus","bicycle","motorcycle"]
SEED = 0

coco = json.load(open(POOL))
id2n = {c["id"]: c["name"] for c in coco["categories"]}
anns_by_img = defaultdict(list)
for a in coco["annotations"]:
    anns_by_img[a["image_id"]].append(a)

# classify + per-image head/tail box counts
info = {}
for im in coco["images"]:
    anns = anns_by_img[im["id"]]
    cs = set(id2n[a["category_id"]] for a in anns)
    n_head_boxes = sum(1 for a in anns if id2n[a["category_id"]] in HEAD)
    n_tail_boxes = sum(1 for a in anns if id2n[a["category_id"]] not in HEAD)
    has_tail = bool(cs - HEAD); has_bike = "bicycle" in cs; is_empty = not cs
    info[im["id"]] = dict(cs=cs, nh=n_head_boxes, nt=n_tail_boxes,
                          has_tail=has_tail, has_bike=has_bike, is_empty=is_empty)

def simulate(headped_ratio, density_cap=None):
    """density_cap: for 'both' images, drop if nh > density_cap * max(nt,1)."""
    random.seed(SEED)
    keep_tb, pool_excess, dropped_density = [], [], 0
    for iid, d in info.items():
        if d['has_tail'] or d['has_bike']:
            if density_cap is not None and d['has_tail'] and d['nh'] > density_cap * max(d['nt'],1):
                dropped_density += 1
                continue
            keep_tb.append(iid)
        else:
            pool_excess.append(iid)
    cap = round(headped_ratio * len(keep_tb))
    ex = pool_excess[:]; random.shuffle(ex)
    kept = set(keep_tb) | set(ex[:cap])

    inst = {k:0 for k in ORDER}; freq = {k:0 for k in ORDER}
    for iid in kept:
        for a in anns_by_img[iid]: inst[id2n[a["category_id"]]] += 1
        for c in info[iid]['cs']: freq[c] += 1
    n = len(kept); tot=sum(inst.values()); tail=sum(inst[k] for k in ORDER if k not in HEAD)
    return n, inst, freq, tot, tail, dropped_density

def report(label, n, inst, freq, tot, tail, dropped=0):
    print(f"\n=== {label}: {n} images, {tot} inst, tail={tail} ({100*tail/tot:.1f}%)"
          + (f", dropped_by_density={dropped}" if dropped else "") + " ===")
    carf = freq['car']/n
    print(f"  {'class':<22}{'inst':>8}{'freq%':>8}{'freq/car':>9}")
    for c in ORDER:
        f = freq[c]/n
        print(f"  {c:<22}{inst[c]:>8}{100*f:>7.1f}%{f/carf:>9.3f}")

print("########## (A) LOWER car/ped-only+empty trim ratio ##########")
for ratio in [0.25, 0.15, 0.10, 0.05, 0.0]:
    n, inst, freq, tot, tail, _ = simulate(ratio)
    report(f"ratio={ratio}", n, inst, freq, tot, tail)

print("\n\n########## (B) ratio=0.10 + density cap on 'both' images ##########")
for dcap in [None, 15, 8, 5]:
    n, inst, freq, tot, tail, dropped = simulate(0.10, dcap)
    report(f"ratio=0.10, density_cap={dcap}", n, inst, freq, tot, tail, dropped)
