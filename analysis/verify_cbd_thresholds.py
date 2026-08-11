"""Step A1 verification: per-class frequencies and ClassBalancedDataset repeat
factors on the curated training set, plus effective images/epoch per threshold.

IMPORTANT: mmengine's ClassBalancedDataset applies math.ceil() to each image's
repeat factor (dataset_wrapper.py: `[dataset_index] * math.ceil(repeat_factor)`).
So only the INTEGER BUCKET matters, not the fractional value:
    r == 1.0        -> 1 copy
    1.0 < r <= 2.0  -> 2 copies
    2.0 < r <= 3.0  -> 3 copies
A class is therefore un-boosted only if thr <= its image frequency.

Class repeat factor  r(c) = max(1, sqrt(thr / f(c)))   f(c) = image-level freq
Image repeat factor  r(i) = max over classes present   (1.0 if image empty)
Effective epoch size = sum of ceil(r(i))

Usage: /home/batashey/miniconda3/envs/lcfusion/bin/python verify_cbd_thresholds.py [ann_file]
"""
import json, os, math, sys
from collections import defaultdict

ROOT = os.path.expanduser("~/mmdetection3d/data/nuscenes_trainval")
ANN  = sys.argv[1] if len(sys.argv) > 1 else "nuscenes_2d_train_curated_v2.coco.json"
HEAD = {"car", "pedestrian", "bicycle"}
ORDER = ["car", "truck", "pedestrian", "traffic_cone", "barrier",
         "trailer", "construction_vehicle", "bus", "bicycle", "motorcycle"]
THRS = [0.17, 0.25, 0.30, 0.34, 0.35, 0.50]

coco = json.load(open(os.path.join(ROOT, ANN)))
id2n = {c["id"]: c["name"] for c in coco["categories"]}

img_cls = {im["id"]: set() for im in coco["images"]}
inst = defaultdict(int)
for a in coco["annotations"]:
    name = id2n[a["category_id"]]
    img_cls[a["image_id"]].add(name)
    inst[name] += 1

n = len(coco["images"])
freq = defaultdict(int)
for cs in img_cls.values():
    for c in cs:
        freq[c] += 1

n_empty = sum(1 for cs in img_cls.values() if not cs)
tot = sum(inst.values())
tail = sum(v for k, v in inst.items() if k not in HEAD)

print(f"=== {ANN} ===")
print(f"images={n}  (empty={n_empty})  instances={tot}  tail={tail} ({100*tail/tot:.1f}%)\n")

hdr = (f"{'class':<22}{'inst':>9}{'freq%':>8}   "
       + "".join(f"{'thr'+str(t):>11}" for t in THRS))
print(hdr); print("-" * len(hdr))
for c in ORDER:
    f = freq[c] / n
    tag = "H" if c in HEAD else "T"
    cells = ""
    for t in THRS:
        r = max(1.0, math.sqrt(t / f))
        cells += f"{r:>6.2f}->{math.ceil(r):<4}"
    print(f"{tag} {c:<20}{inst[c]:>9}{100*f:>7.1f}%   {cells}")

print("\n" + "=" * 78)
for t in THRS:
    rf = {c: max(1.0, math.sqrt(t / (freq[c] / n))) for c in ORDER}
    eff = 0
    buckets = defaultdict(int)
    ped_driven = 0
    for cs in img_cls.values():
        if cs:
            r = max(rf[c] for c in cs)
            # would the factor be lower if pedestrian were excluded?
            others = [rf[c] for c in cs if c != "pedestrian"]
            if "pedestrian" in cs and math.ceil(r) > math.ceil(max(others, default=1.0)):
                ped_driven += 1
        else:
            r = 1.0
        k = math.ceil(r)
        eff += k
        buckets[k] += 1
    head_un = [c for c in ORDER if c in HEAD and freq[c]/n >= t]
    print(f"thr={t:<5} effective={eff:>7} ({eff/n:.2f}x)   "
          f"copies: " + " ".join(f"{k}x:{v}" for k, v in sorted(buckets.items()))
          + f"   head classes at 1x: {','.join(head_un) if head_un else 'NONE'}"
          + (f"   ped-driven imgs: {ped_driven}" if ped_driven else ""))

print("\nClean band = thresholds where car, truck AND pedestrian all stay at 1 copy")
print(f"  requires thr <= pedestrian freq = {freq['pedestrian']/n:.3f}")
print(f"  and to give bicycle/motorcycle 3 copies requires thr > 4*moto_freq = "
      f"{4*freq['motorcycle']/n:.3f}")
