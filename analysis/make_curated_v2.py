"""Refined curation (v2): keep every tail-containing, bicycle-containing and
EMPTY image (empties = background negatives, since DINO base uses
filter_empty_gt=False); drop ALL car/pedestrian-only images.
Reads the full pool json, writes nuscenes_2d_train_curated_v2.coco.json."""
import json, os, math
from collections import defaultdict

ROOT = os.path.expanduser("~/mmdetection3d/data/nuscenes_trainval")
POOL = os.path.join(ROOT, "nuscenes_2d_train_pool.coco.json")
OUT  = os.path.join(ROOT, "nuscenes_2d_train_curated_v2.coco.json")
HEAD = {"car", "pedestrian", "bicycle"}
ORDER = ["car","pedestrian","barrier","traffic_cone","truck","trailer",
         "construction_vehicle","bus","bicycle","motorcycle"]

coco = json.load(open(POOL))
id2n = {c["id"]: c["name"] for c in coco["categories"]}
anns_by_img = defaultdict(list)
for a in coco["annotations"]:
    anns_by_img[a["image_id"]].append(a)

keep = set()
for im in coco["images"]:
    cs = set(id2n[a["category_id"]] for a in anns_by_img[im["id"]])
    has_tail = bool(cs - HEAD)
    has_bike = "bicycle" in cs
    is_empty = not cs
    is_carped_only = (not has_tail) and (not has_bike) and (not is_empty)
    if not is_carped_only:          # drop only pure car/ped images
        keep.add(im["id"])

out = dict(images=[im for im in coco["images"] if im["id"] in keep],
           annotations=[], categories=coco["categories"])
aid = 0
for a in coco["annotations"]:
    if a["image_id"] in keep:
        a = dict(a); a["id"] = aid; aid += 1; out["annotations"].append(a)
json.dump(out, open(OUT, "w"))

# report
inst = {k:0 for k in ORDER}; freq = {k:0 for k in ORDER}
ho=to=bo=em=0
for im in out["images"]:
    cs = set(id2n[a["category_id"]] for a in anns_by_img[im["id"]])
    for a in anns_by_img[im["id"]]: inst[id2n[a["category_id"]]] += 1
    for c in cs: freq[c] += 1
    ht=bool(cs&HEAD); tt=bool(cs-HEAD)
    if not cs: em+=1
    elif ht and tt: bo+=1
    elif ht: ho+=1
    else: to+=1
n=len(out["images"]); tot=sum(inst.values()); tail=sum(inst[k] for k in ORDER if k not in HEAD)
print(f"===== CURATED v2: {n} images, {tot} inst, tail={tail} ({100*tail/tot:.1f}%) =====")
print(f"  head-only(=bicycle-only)={ho} tail-only={to} both={bo} empty={em}")
print(f"  {'class':<21}{'inst':>9}{'share%':>8}{'imgfreq':>9}{'freq%':>8}{'r@0.17':>8}")
for k in ORDER:
    f=freq[k]/n; r=max(1.0,math.sqrt(0.17/f)) if f>0 else 0
    tag='H' if k in HEAD else 'T'
    print(f"{tag} {k:<19}{inst[k]:>9}{100*inst[k]/tot:>7.1f}%{freq[k]:>9}{100*f:>7.1f}%{r:>8.2f}")
print(f"  car:motorcycle = {inst['car']/inst['motorcycle']:.1f}x")
print(f"\nwrote {n} imgs, {len(out['annotations'])} anns -> {OUT}")
