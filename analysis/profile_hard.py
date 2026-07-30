"""Profile the 3 hardest classes (barrier, construction_vehicle, trailer)
in the original part-1 train vs the curated train set."""
import json, os, numpy as np
from collections import defaultdict

ROOT = os.path.expanduser("~/mmdetection3d/data/nuscenes_trainval")
HARD = ["barrier", "construction_vehicle", "trailer"]
HEAD = {"car", "pedestrian", "bicycle"}

def load(fn):
    c = json.load(open(os.path.join(ROOT, fn)))
    id2n = {x["id"]: x["name"] for x in c["categories"]}
    imgcls = defaultdict(set)
    areas = defaultdict(list)
    inst = defaultdict(int)
    for a in c["annotations"]:
        n = id2n[a["category_id"]]
        imgcls[a["image_id"]].add(n)
        inst[n] += 1
        areas[n].append(a["bbox"][2] * a["bbox"][3])
    return c, imgcls, areas, inst, len(c["images"])

def size_split(areas):
    a = np.array(areas)
    s = (a < 32**2).mean() * 100
    m = ((a >= 32**2) & (a < 96**2)).mean() * 100
    l = (a >= 96**2).mean() * 100
    return s, m, l, np.median(a)

for fn, tag in [("nuscenes_2d_train.coco.json", "ORIGINAL part-1"),
                ("nuscenes_2d_train_curated_v2.coco.json", "CURATED v2")]:
    c, imgcls, areas, inst, nimg = load(fn)
    imgfreq = defaultdict(int)
    for s in imgcls.values():
        for n in s: imgfreq[n] += 1
    print(f"\n===== {tag}: {nimg} images =====")
    print(f"  {'class':<22}{'imgs':>7}{'imgfreq%':>9}{'inst':>8}{'inst/img':>9}"
          f"{'%small':>8}{'%med':>7}{'%large':>8}{'medArea':>9}")
    for n in HARD:
        s,m,l,med = size_split(areas[n])
        print(f"  {n:<22}{imgfreq[n]:>7}{100*imgfreq[n]/nimg:>8.1f}%{inst[n]:>8}"
              f"{inst[n]/imgfreq[n]:>9.2f}{s:>7.0f}%{m:>6.0f}%{l:>7.0f}%{med:>9.0f}")
    # images containing ANY / co-occurrence among the 3
    any3 = sum(1 for s in imgcls.values() if s & set(HARD))
    multi = sum(1 for s in imgcls.values() if len(s & set(HARD)) >= 2)
    # how often each hard class shares an image with a head class
    print(f"  images with >=1 of the 3 hard classes: {any3} ({100*any3/nimg:.1f}%)")
    print(f"  images with >=2 of the 3 hard classes: {multi}")
    for n in HARD:
        with_head = sum(1 for s in imgcls.values() if n in s and (s & HEAD))
        print(f"    {n}: co-occurs with a head class in {with_head}/{imgfreq[n]} "
              f"({100*with_head/imgfreq[n]:.0f}%) of its images")
