"""Map downloaded parts 1/2/3 keyframes to scenes and compute 3D per-class
distribution (sample-level) using the nuScenes devkit metadata."""
import os, json
from collections import defaultdict
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.utils import category_to_detection_name

ROOT = os.path.expanduser("~/mmdetection3d/data/nuscenes_trainval")
HEAD = {"car", "pedestrian", "bicycle"}
ORDER = ["car","truck","construction_vehicle","bus","trailer","barrier",
         "motorcycle","bicycle","pedestrian","traffic_cone"]

nusc = NuScenes(version="v1.0-trainval", dataroot=ROOT, verbose=False)

# filename -> sample_token for LIDAR_TOP keyframes
fn2sample = {}
for sd in nusc.sample_data:
    if sd['channel'] == 'LIDAR_TOP' and sd['is_key_frame']:
        fn2sample[sd['filename']] = sd['sample_token']

def load_part(path):
    toks = set()
    for line in open(path):
        fn = line.strip()
        if fn in fn2sample:
            toks.add(fn2sample[fn])
    return toks

# part1 = extracted (present on disk); parts 2/3 from tarball listings
part_files = {
    "part1": None,  # from disk
    "part2": "/tmp/part2_lidar.txt",
    "part3": "/tmp/part3_lidar.txt",
}

# part1 present samples: filenames on disk
part1_toks = set()
p1dir = os.path.join(ROOT, "samples/LIDAR_TOP")
for f in os.listdir(p1dir):
    fn = "samples/LIDAR_TOP/" + f
    if fn in fn2sample:
        part1_toks.add(fn2sample[fn])

parts = {"part1": part1_toks,
         "part2": load_part("/tmp/part2_lidar.txt"),
         "part3": load_part("/tmp/part3_lidar.txt")}

# sample_token -> scene_token
def scene_of(stok):
    return nusc.get('sample', stok)['scene_token']

# official split membership
from nuscenes.utils import splits
train_scenes = set(splits.train)
val_scenes   = set(splits.val)

def analyze(sample_toks, restrict_split=None):
    inst = defaultdict(int)
    comp = defaultdict(int)  # head-only/tail-only/both/empty
    n_samples = 0
    scenes = set()
    for stok in sample_toks:
        sc = nusc.get('scene', scene_of(stok))
        scenes.add(sc['name'])
        if restrict_split == 'train' and sc['name'] not in train_scenes:
            continue
        if restrict_split == 'val' and sc['name'] not in val_scenes:
            continue
        n_samples += 1
        s = nusc.get('sample', stok)
        classes = set()
        for atok in s['anns']:
            ann = nusc.get('sample_annotation', atok)
            det = category_to_detection_name(ann['category_name'])
            if det is None:
                continue
            inst[det] += 1
            classes.add(det)
        has_head = bool(classes & HEAD)
        has_tail = bool(classes - HEAD)
        if not classes: comp['empty'] += 1
        elif has_head and has_tail: comp['both'] += 1
        elif has_head: comp['head_only'] += 1
        else: comp['tail_only'] += 1
    return inst, comp, n_samples, scenes

print(f"{'metadata loaded; keyframe map size':<40}{len(fn2sample)}")
for p in ["part1","part2","part3"]:
    toks = parts[p]
    inst, comp, n, scenes = analyze(toks)
    # split breakdown of scenes
    n_train = sum(1 for sc in scenes if sc in train_scenes)
    n_val   = sum(1 for sc in scenes if sc in val_scenes)
    tail = sum(v for k,v in inst.items() if k not in HEAD)
    tot  = sum(inst.values())
    print(f"\n===== {p}: {n} samples, {len(scenes)} scenes "
          f"({n_train} train / {n_val} val) =====")
    print(f"  composition: head_only={comp['head_only']} tail_only={comp['tail_only']} "
          f"both={comp['both']} empty={comp['empty']}")
    print(f"  instances={tot}  tail_instances={tail} ({100*tail/tot:.1f}%)")
    print(f"  {'class':<22}{'instances':>10}")
    for n_ in ORDER:
        print(f"  {n_:<22}{inst[n_]:>10}")

# TRAIN-ONLY pooled analysis: what parts 2+3 add to the TRAIN set
print("\n\n########## TRAIN-SCENE-ONLY yield (what can be added to training) ##########")
for label, toks in [("part1_train", parts["part1"]),
                    ("part2_train", parts["part2"]),
                    ("part3_train", parts["part3"]),
                    ("part2+3_train", parts["part2"] | parts["part3"])]:
    inst, comp, n, scenes = analyze(toks, restrict_split='train')
    tail = sum(v for k,v in inst.items() if k not in HEAD)
    tot  = sum(inst.values()) or 1
    print(f"\n=== {label}: {n} train samples ===")
    print(f"  comp: head_only={comp['head_only']} tail_only={comp['tail_only']} both={comp['both']} empty={comp['empty']}")
    print(f"  tail_instances={tail} / {tot} ({100*tail/tot:.1f}%)")
    line = "  ".join(f"{n_}={inst[n_]}" for n_ in ORDER)
    print("  "+line)
