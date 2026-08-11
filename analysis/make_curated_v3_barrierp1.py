"""Curated v3: restricts BARRIER'S TRAINING DATA VOLUME to part-1 only (same
exposure as Run C), while every other tail class gets the full curated
multi-part benefit (parts 1,2,3,5,6).

Different from curated_cbd030_nobarrier.py's ablation, which only excluded
barrier from ClassBalancedDataset's oversampling REPETITION but still trained
on all 75,529 barrier boxes from every part. That ablation did not recover
barrier (0.438->0.449). This tests the other half of the hypothesis: maybe
it's the raw VOLUME/diversity of barrier training data from parts 2,3,5,6
(not the oversampling repetition of it) that shifts barrier's confidence
calibration, since barrier's 2D AP was unaffected (0.187 vs 0.180) but its
matched-2D-score dropped sharply (0.634->0.454) purely from more barrier
training exposure.

Policy (starting from the curated_v2 policy: keep every tail/bicycle/empty
image, drop pure car/ped-only images), with one addition:
  * images whose ONLY tail class is barrier: keep ONLY if the image is
    also present in the part-1-only train set (nuscenes_2d_train.coco.json,
    identified by file_name). Drop if sourced from parts 2/3/5/6.
  * images containing barrier AND another tail class: ALWAYS kept regardless
    of source part (dropping them would also lose the other tail class's
    signal, which we want to preserve at full multi-part volume).
  * every other rule from curated_v2 unchanged (tail-only, bicycle-only,
    empty images from ANY part kept; car/ped-only from any part dropped).

Reads the full pool json (parts 1,2,3,5,6). Writes
nuscenes_2d_train_curated_v3_barrierp1.coco.json.
"""
import json, os, math
from collections import defaultdict

ROOT = os.path.expanduser("~/mmdetection3d/data/nuscenes_trainval")
POOL = os.path.join(ROOT, "nuscenes_2d_train_pool.coco.json")
PART1_TRAIN = os.path.join(ROOT, "nuscenes_2d_train.coco.json")
OUT = os.path.join(ROOT, "nuscenes_2d_train_curated_v3_barrierp1.coco.json")
HEAD = {"car", "pedestrian", "bicycle"}
ORDER = ["car", "pedestrian", "barrier", "traffic_cone", "truck", "trailer",
         "construction_vehicle", "bus", "bicycle", "motorcycle"]

part1 = json.load(open(PART1_TRAIN))
part1_files = set(im["file_name"] for im in part1["images"])
print(f"part-1 train images (reference set): {len(part1_files)}")

coco = json.load(open(POOL))
id2n = {c["id"]: c["name"] for c in coco["categories"]}
anns_by_img = defaultdict(list)
for a in coco["annotations"]:
    anns_by_img[a["image_id"]].append(a)

img_by_id = {im["id"]: im for im in coco["images"]}

keep = set()
dropped_barrier_only_multipart = 0
for im in coco["images"]:
    cs = set(id2n[a["category_id"]] for a in anns_by_img[im["id"]])
    has_tail = bool(cs - HEAD)
    has_bike = "bicycle" in cs
    is_empty = not cs
    is_carped_only = (not has_tail) and (not has_bike) and (not is_empty)

    if is_carped_only:
        continue  # drop pure car/ped images, same as curated_v2

    # NEW rule: images whose ONLY tail class is barrier
    tail_classes = cs - HEAD
    is_barrier_only_tail = (tail_classes == {"barrier"})
    if is_barrier_only_tail:
        is_part1 = im["file_name"] in part1_files
        if not is_part1:
            dropped_barrier_only_multipart += 1
            continue  # drop: barrier-only image sourced from parts 2/3/5/6

    keep.add(im["id"])

out = dict(images=[im for im in coco["images"] if im["id"] in keep],
           annotations=[], categories=coco["categories"])
aid = 0
for a in coco["annotations"]:
    if a["image_id"] in keep:
        a = dict(a); a["id"] = aid; aid += 1
        out["annotations"].append(a)
json.dump(out, open(OUT, "w"))

# report
inst = {k: 0 for k in ORDER}
freq = {k: 0 for k in ORDER}
ho = to = bo = em = 0
for im in out["images"]:
    cs = set(id2n[a["category_id"]] for a in anns_by_img[im["id"]])
    for a in anns_by_img[im["id"]]:
        inst[id2n[a["category_id"]]] += 1
    for c in cs:
        freq[c] += 1
    ht = bool(cs & HEAD)
    tt = bool(cs - HEAD)
    if not cs:
        em += 1
    elif ht and tt:
        bo += 1
    elif ht:
        ho += 1
    else:
        to += 1

n = len(out["images"])
tot = sum(inst.values())
tail = sum(inst[k] for k in ORDER if k not in HEAD)
print(f"\ndropped {dropped_barrier_only_multipart} barrier-only images sourced from parts 2/3/5/6")
print(f"\n===== CURATED v3 (barrier restricted to part-1): {n} images, {tot} inst, "
      f"tail={tail} ({100*tail/tot:.1f}%) =====")
print(f"  head-only(=bicycle-only)={ho} tail-only={to} both={bo} empty={em}")
print(f"  {'class':<21}{'inst':>9}{'share%':>8}{'imgfreq':>9}{'freq%':>8}{'r@0.30':>8}")
for k in ORDER:
    f = freq[k] / n
    r = max(1.0, math.sqrt(0.30 / f)) if f > 0 else 0
    tag = 'H' if k in HEAD else 'T'
    print(f"{tag} {k:<19}{inst[k]:>9}{100*inst[k]/tot:>7.1f}%{freq[k]:>9}{100*f:>7.1f}%{r:>8.2f}")
print(f"\nbarrier instance count vs Run C's 18,170: {inst['barrier']}"
      f" ({100*inst['barrier']/18170:.0f}% of RunC's barrier volume)")
print(f"\nwrote {n} imgs, {len(out['annotations'])} anns -> {OUT}")
