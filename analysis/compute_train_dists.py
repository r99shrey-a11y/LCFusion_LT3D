"""Per-class training-set distributions for Run C (base + effective oversampled
at thr=0.17), curated (part-1), and curated_v2 (parts 1,2,3,5,6)."""
import json, os, math
from collections import defaultdict

ROOT = os.path.expanduser("~/mmdetection3d/data/nuscenes_trainval")
ORDER = ["car","pedestrian","barrier","traffic_cone","truck",
         "trailer","construction_vehicle","bus","bicycle","motorcycle"]

def load(fn):
    c = json.load(open(os.path.join(ROOT, fn)))
    id2n = {x["id"]: x["name"] for x in c["categories"]}
    per_img = defaultdict(list)
    for a in c["annotations"]:
        per_img[a["image_id"]].append(id2n[a["category_id"]])
    return c["images"], per_img

def basic(fn):
    imgs, per_img = load(fn)
    inst = defaultdict(int); imgf = defaultdict(int)
    for im in imgs:
        cs = per_img.get(im["id"], [])
        for x in cs: inst[x] += 1
        for x in set(cs): imgf[x] += 1
    return len(imgs), inst, imgf

def runc_effective(fn, thr=0.17):
    imgs, per_img = load(fn)
    N = len(imgs)
    imgf = defaultdict(int)
    for im in imgs:
        for x in set(per_img.get(im["id"], [])): imgf[x] += 1
    rf = {c: max(1.0, math.sqrt(thr / (imgf[c]/N))) for c in imgf}
    eff_inst = defaultdict(float); eff_imgf = defaultdict(float); eff_N = 0.0
    for im in imgs:
        cs = per_img.get(im["id"], [])
        r = max([rf[c] for c in set(cs)], default=1.0)
        eff_N += r
        for x in cs: eff_inst[x] += r
        for x in set(cs): eff_imgf[x] += r
    return eff_N, eff_inst, eff_imgf

# available train files
avail = {
  "RunC_base(part1)":        "nuscenes_2d_train.coco.json",
  "curated_v1(p1-3,r0.25)":  "nuscenes_2d_train_curated.coco.json",
  "curated_v2(p1,2,3,5,6)":  "nuscenes_2d_train_curated_v2.coco.json",
}
res = {}
for tag, fn in avail.items():
    if os.path.exists(os.path.join(ROOT, fn)):
        res[tag] = basic(fn)

effN, effI, effF = runc_effective("nuscenes_2d_train.coco.json")

print("=== IMAGES ===")
for tag,(n,_,_) in res.items(): print(f"  {tag:<26} {n}")
print(f"  RunC_EFFECTIVE(oversampled)  {effN:.0f}")

print("\n=== per-class INSTANCES ===")
hdr = f"{'class':<22}" + "".join(f"{t.split('(')[0]:>14}" for t in res) + f"{'RunC_eff':>12}"
print(hdr)
for c in ORDER:
    row = f"{c:<22}"
    for tag,(n,inst,imgf) in res.items(): row += f"{inst[c]:>14}"
    row += f"{effI[c]:>12.0f}"
    print(row)

print("\n=== per-class IMAGE-FREQUENCY %% ===")
print(f"{'class':<22}" + "".join(f"{t.split('(')[0]:>14}" for t in res) + f"{'RunC_eff':>12}")
for c in ORDER:
    row = f"{c:<22}"
    for tag,(n,inst,imgf) in res.items(): row += f"{100*imgf[c]/n:>13.1f}%"
    row += f"{100*effF[c]/effN:>11.1f}%"
    print(row)
