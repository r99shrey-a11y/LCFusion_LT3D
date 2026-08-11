"""Diagnostic: compare barrier GROUND-TRUTH label statistics between part-1
train and the multi-part curated_v2 train set. Looks for a label-quality
shift (box size, aspect ratio, position, density per image) that could
explain why barrier's 2D/fused quality degrades with more parts."""
import json, os
from collections import defaultdict

ROOT = os.path.expanduser("~/mmdetection3d/data/nuscenes_trainval")

def barrier_stats(fn, label):
    coco = json.load(open(os.path.join(ROOT, fn)))
    id2n = {c["id"]: c["name"] for c in coco["categories"]}
    bid = [k for k, v in id2n.items() if v == "barrier"][0]

    anns = [a for a in coco["annotations"] if a["category_id"] == bid]
    n = len(anns)
    if n == 0:
        print(f"{label}: no barrier annotations"); return

    ws = sorted(a["bbox"][2] for a in anns)
    hs = sorted(a["bbox"][3] for a in anns)
    areas = sorted(a["bbox"][2] * a["bbox"][3] for a in anns)
    aspects = sorted(a["bbox"][2] / max(a["bbox"][3], 1e-6) for a in anns)

    def pct(arr, p):
        i = min(len(arr) - 1, int(p / 100 * len(arr)))
        return arr[i]

    # per-image barrier count (density) + truncation rate
    per_img = defaultdict(int)
    trunc = 0
    W, H = 1600, 900
    for a in anns:
        per_img[a["image_id"]] += 1
        x, y, w, h = a["bbox"]
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            trunc += 1

    n_imgs = len(per_img)
    densities = sorted(per_img.values())

    print(f"\n=== {label}: {n} barrier boxes across {n_imgs} images ===")
    print(f"  width  px : p10={pct(ws,10):.0f} p50={pct(ws,50):.0f} p90={pct(ws,90):.0f}")
    print(f"  height px : p10={pct(hs,10):.0f} p50={pct(hs,50):.0f} p90={pct(hs,90):.0f}")
    print(f"  area   px2: p10={pct(areas,10):.0f} p50={pct(areas,50):.0f} p90={pct(areas,90):.0f}")
    print(f"  aspect w/h: p10={pct(aspects,10):.2f} p50={pct(aspects,50):.2f} p90={pct(aspects,90):.2f}")
    print(f"  truncated (touches image border): {trunc} ({100*trunc/n:.0f}%)")
    print(f"  boxes/image: p50={pct(densities,50)} p90={pct(densities,90)} max={densities[-1]}")
    small = sum(1 for a in areas if a < 32**2)
    print(f"  tiny boxes (<32x32 px area): {small} ({100*small/n:.0f}%)")

print("Comparing barrier GT label statistics: part-1 vs multi-part curated_v2\n")
barrier_stats("nuscenes_2d_train.coco.json", "PART-1 train (Run C's data)")
barrier_stats("nuscenes_2d_train_curated_v2.coco.json", "CURATED_V2 train (parts 1,2,3,5,6)")
