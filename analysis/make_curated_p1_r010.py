"""Curated train set restricted to PART-1 scenes only, ratio=0.10 (improved
trim), for a same-data comparison against Run C (which also trains on
part-1-only nuscenes_2d_train.coco.json)."""
import json, os, random
from collections import defaultdict

ROOT = os.path.expanduser("~/mmdetection3d/data/nuscenes_trainval")
POOL = os.path.join(ROOT, "nuscenes_2d_train_pool.coco.json")   # full pool, all parts
PART1_BASE = os.path.join(ROOT, "nuscenes_2d_train.coco.json")  # part-1-only original
OUT  = os.path.join(ROOT, "nuscenes_2d_train_curated_p1_r010.coco.json")
HEAD = {"car", "pedestrian", "bicycle"}
PRESERVE = {"bicycle"}
RATIO = 0.10
SEED = 0

# Restrict to part-1 images: use the file_name set already in the part-1-only json
part1 = json.load(open(PART1_BASE))
part1_files = set(im["file_name"] for im in part1["images"])

pool = json.load(open(POOL))
id2n = {c["id"]: c["name"] for c in pool["categories"]}
anns_by_img = defaultdict(list)
for a in pool["annotations"]:
    anns_by_img[a["image_id"]].append(a)

p1_images = [im for im in pool["images"] if im["file_name"] in part1_files]
print(f"part-1 images found in pool: {len(p1_images)} (part-1 original: {len(part1['images'])})")

info = {}
for im in p1_images:
    anns = anns_by_img[im["id"]]
    cs = set(id2n[a["category_id"]] for a in anns)
    info[im["id"]] = dict(cs=cs, has_tail=bool(cs-HEAD), has_bike="bicycle" in cs, is_empty=not cs)

keep_tb = [im["id"] for im in p1_images if info[im["id"]]['has_tail'] or info[im["id"]]['has_bike']]
pool_excess = [im["id"] for im in p1_images if not (info[im["id"]]['has_tail'] or info[im["id"]]['has_bike'])]

random.seed(SEED)
cap = round(RATIO * len(keep_tb))
random.shuffle(pool_excess)
kept = set(keep_tb) | set(pool_excess[:cap])

out = dict(images=[im for im in p1_images if im["id"] in kept],
           annotations=[], categories=pool["categories"])
aid = 0
for iid in kept:
    for a in anns_by_img[iid]:
        a = dict(a); a["id"] = aid; aid += 1; out["annotations"].append(a)
json.dump(out, open(OUT, "w"))

# report
ORDER = ["car","pedestrian","barrier","traffic_cone","truck","trailer",
         "construction_vehicle","bus","bicycle","motorcycle"]
inst = defaultdict(int); freq = defaultdict(int)
for iid in kept:
    for a in anns_by_img[iid]: inst[id2n[a["category_id"]]] += 1
    for c in info[iid]['cs']: freq[c] += 1
n = len(kept); tot = sum(inst.values()); tail = sum(inst[k] for k in ORDER if k not in HEAD)
print(f"\n=== part-1 curated (ratio={RATIO}): {n} images, {tot} inst, tail={tail} ({100*tail/tot:.1f}%) ===")
carf = freq['car']/n
for c in ORDER:
    f = freq[c]/n
    print(f"  {c:<22}{inst[c]:>8}{100*f:>7.1f}%  freq/car={f/carf:.3f}")
print(f"\nwrote {n} images, {len(out['annotations'])} anns -> {OUT}")
