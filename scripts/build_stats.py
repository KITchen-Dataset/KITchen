#!/usr/bin/env python3
"""Build data/stats.json for the website from a KITchen release folder.

Re-run this whenever the release changes (new pose version, new labels):

    python scripts/build_stats.py /path/to/kat_release --split-file splits/setups.csv --out data/stats.json

Expected layout (one folder per release part; each holds many recording sequences)::

    kat_release/
      KITchen_labels_<N>/
        classes.txt        class names, line number = class id = BOP obj_id
        labels/<stem>.txt  YOLO boxes: "cls cx cy w h" (normalised)
        ignore.json        {stem: [row indices not in the confident set]}
        scene_gt.json      {stem: [{obj_id, cam_R_m2c, cam_t_m2c, ...}]}  (optional)

The train / validation / test numbers follow the KITchen benchmark split
(kitchenbench v1.0): train is the lab room, validation and test are the
built-in kitchen, with each kitchen setup (one arrangement of objects) kept
whole in one split. The split comes from ``splits/setups.csv`` (stem,setup,split)
in the benchmark archive, or from a run list (start,end,setup,split; stems
inclusive); frames it does not list are left out, as in kitchenbench build.

Only the standard library is used, so it runs anywhere.
"""
from __future__ import annotations

import argparse
import json
import csv
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path

GRID_W, GRID_H = 48, 27          # heat-map resolution for box centres (16:9)
DIST_BIN_CM = 10                 # camera-distance histogram bin width
DIST_MAX_CM = 400
OBJ_PER_FRAME_MAX = 80           # last bin holds 80+ objects
AREA_BINS = [0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0]  # fraction of image
BENCH_SUBSETS = [("train", "Train"), ("val", "Validation"), ("test", "Test"), ("buffer", "Unassigned")]

# Coarse groups for the class browser.  Classes not listed fall into "Packaged food".
CATEGORY_RULES = [
    ("YCB objects", r"^(master_chef_can|cracker_box|sugar_box|tomato_soup_can|mustard_bottle|"
                    r"tuna_fish_can|pudding_box|gelatin_box|potted_meat_can|banana|strawberry|apple|"
                    r"lemon|peach|pear|orange|plum|pitcher_base|bowl|mug|skillet|spatula|chips_can)$"),
    ("Cups", r"(_cups|cup_large|green-cup|GreenCup|FlowerCup)$"),
    ("Kitchenware", r"^(mixing_bowl_green|mixing_bowl_small|frying-pan|plate|egg_whisk|ladle|salad_spoon|"
                    r"salad_fork|tablespoon|cutting_board|knife_black|draining_rack|senseo-coffee-machine)$"),
    ("Drinks", r"(milk|Milk(?!Rice)|drink|Drink|juice|tea|Tea|Coke|IceCoffee)"),
]
_CATEGORY_RES = [(name, re.compile(rx)) for name, rx in CATEGORY_RULES]


def category(name: str) -> str:
    if name == "other":
        return "Other"
    for cat, rx in _CATEGORY_RES:
        if rx.search(name):
            return cat
    return "Packaged food"


def pretty(name: str) -> str:
    s = re.sub(r"[_\-]+", " ", name)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s)
    s = re.sub(r"Potatoe(?= [A-Z])", "Potato", s).replace("Knaeckebrot", "Knäckebrot").replace("Org ", "Organic ").replace("Hering", "Herring")
    return s[:1].upper() + s[1:]


def read_split_file(path: Path) -> dict[str, tuple[str, str]]:
    """{stem: (setup, split)}; the same checks as kitchenbench.build.read_split_file."""
    rows, subs = {}, {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"] not in ("train", "val", "test", "buffer"):
                raise SystemExit(f"{path}: unknown split {r['split']!r} for {r.get('stem', r.get('start'))}")
            if "stem" in r:
                stems = [r["stem"]]
            else:                # run list: start..end inclusive
                w = len(r["start"])
                stems = [str(n).zfill(w) for n in range(int(r["start"]), int(r["end"]) + 1)]
            for stem in stems:
                if stem in rows:
                    raise SystemExit(f"{path}: {stem} listed twice")
                rows[stem] = (r["setup"], r["split"])
            if r["split"] != "buffer":
                subs.setdefault(r["setup"], set()).add(r["split"])
    mixed = sorted(k for k, v in subs.items() if len(v) > 1)
    if mixed:
        raise SystemExit(f"{path}: setups in more than one split: {mixed[:5]}")
    return rows


def seq_dirs(root: Path) -> list[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir() and re.search(r"_(\d+)$", p.name) and (p / "labels").is_dir()]
    return sorted(dirs, key=lambda p: int(re.search(r"_(\d+)$", p.name).group(1)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("release", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "stats.json")
    bench = ap.add_argument_group("benchmark split (keep in sync with kitchenbench build)")
    bench.add_argument("--split-file", type=Path, required=True, help="splits/setups.csv from the benchmark archive, or its run list")
    args = ap.parse_args()
    setups = read_split_file(args.split_file)

    seqs = seq_dirs(args.release)
    if not seqs:
        raise SystemExit(f"no KITchen_*_<N>/labels folders under {args.release}")
    classes = [c for c in (seqs[0] / "classes.txt").read_text().splitlines() if c.strip()]
    n_cls = len(classes)

    inst = Counter(); conf = Counter(); pose_cnt = Counter()
    per_frame = Counter(); heat = [[0] * GRID_W for _ in range(GRID_H)]
    area_hist = [0] * (len(AREA_BINS) - 1)
    dist_hist = [0] * (DIST_MAX_CM // DIST_BIN_CM)
    dist_by_cls: dict[int, list[float]] = {}
    seq_rows = []; timeline = []
    sub_rows = {k: {"id": k, "name": n, "objects": 0, "confident": 0, "poses": 0, "setups": set()} for k, n in BENCH_SUBSETS}
    bands = []                   # [start, end) runs of one subset, in timeline (frame) order
    tot_frames = tot_obj = tot_conf = tot_pose = 0

    for sd in seqs:
        sid = int(re.search(r"_(\d+)$", sd.name).group(1))
        ignore = json.loads((sd / "ignore.json").read_text()) if (sd / "ignore.json").exists() else {}
        files = sorted((sd / "labels").glob("*.txt"))
        subset = {f.stem: setups[f.stem][1] if f.stem in setups else "buffer" for f in files}
        s_obj = s_conf = 0
        for f in files:
            stem = f.stem
            bad = set(ignore.get(stem, []))
            rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
            sub = sub_rows[subset[stem]]
            if stem in setups:
                sub["setups"].add(setups[stem][0])
            if bands and bands[-1][2] == sub["id"]:
                bands[-1][1] += 1
            else:
                bands.append([len(timeline), len(timeline) + 1, sub["id"]])
            for i, r in enumerate(rows):
                c = int(r[0]); cx, cy, w, h = map(float, r[1:5])
                inst[c] += 1
                if i not in bad:
                    conf[c] += 1; s_conf += 1; sub["confident"] += 1
                heat[min(GRID_H - 1, max(0, int(cy * GRID_H)))][min(GRID_W - 1, max(0, int(cx * GRID_W)))] += 1
                a = w * h
                for b in range(len(area_hist)):
                    if a < AREA_BINS[b + 1] or b == len(area_hist) - 1:
                        area_hist[b] += 1; break
            per_frame[min(len(rows), OBJ_PER_FRAME_MAX)] += 1
            s_obj += len(rows); sub["objects"] += len(rows)
            timeline.append(len(rows))

        s_pose = 0
        gt_path = sd / "scene_gt.json"
        if gt_path.exists():
            gt = json.loads(gt_path.read_text())
            unlabelled = [s for s in gt if s not in subset]
            if unlabelled:
                print(f"warning: {sd.name}/scene_gt.json has {len(unlabelled)} frames without labels; skipped")
            for stem, objs in gt.items():
                if stem not in subset:
                    continue
                sub = sub_rows[subset[stem]]
                for o in objs:
                    c = int(o["obj_id"]); pose_cnt[c] += 1; s_pose += 1; sub["poses"] += 1
                    t = o["cam_t_m2c"]
                    d = math.sqrt(sum(v * v for v in t))
                    d_cm = d * 100 if d < 20 else d / 10       # metres (old files) or millimetres (BOP)
                    dist_hist[min(len(dist_hist) - 1, int(d_cm // DIST_BIN_CM))] += 1
                    dist_by_cls.setdefault(c, []).append(d_cm)
            del gt

        seq_rows.append({"id": sid, "name": f"Part {sid}", "objects": s_obj, "confident": s_conf, "poses": s_pose})
        tot_frames += len(files); tot_obj += s_obj; tot_conf += s_conf; tot_pose += s_pose

    # Down-sample the per-frame object count into a density strip (~600 points).
    step = max(1, len(timeline) // 600)
    strip = [round(sum(timeline[i:i + step]) / len(timeline[i:i + step]), 1) for i in range(0, len(timeline), step)]

    def median(v):
        v = sorted(v); n = len(v)
        return None if not n else (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2)

    cls_rows = []
    for c, name in enumerate(classes):
        dl = dist_by_cls.get(c, [])
        cls_rows.append({"id": c, "key": name, "name": pretty(name), "category": category(name),
                         "instances": inst[c], "confident": conf[c],
                         "poses": pose_cnt[c], "median_dist_cm": round(median(dl)) if dl else None,
                         "model": f"obj_{c:06d}" if c > 0 else None})

    out = {
        "generated": date.today().isoformat(),
        "image_size": [1920, 1080],
        "camera": "Azure Kinect (RGB-D), mounted on the ARMAR humanoid's head",
        # Object counts only: the page shows no frame counts, so none are written here.
        "totals": {"objects": tot_obj, "confident": tot_conf,
                   "classes": n_cls - (1 if classes and classes[0] == "other" else 0),
                   "parts": len(seqs), "poses": tot_pose,
                   "objects_per_frame_mean": round(tot_obj / max(1, tot_frames), 1)},
        "parts": seq_rows,
        "benchmark": {"unit": "setup",
                      "splits": [{k: v for k, v in r.items() if k != "setups"} for r in sub_rows.values()],
                      # [start, end) of each run as a fraction of the recording order
                      "bands": [[round(a / tot_frames, 4), round(b / tot_frames, 4), s] for a, b, s in bands]},
        "classes": cls_rows,
        # share of images holding k objects (last bin: OBJ_PER_FRAME_MAX or more)
        "objects_per_frame": [round(per_frame[k] / max(1, tot_frames), 4) for k in range(OBJ_PER_FRAME_MAX + 1)],
        "box_area_bins": AREA_BINS, "box_area_hist": area_hist,
        "distance_bin_cm": DIST_BIN_CM, "distance_hist": dist_hist,
        "center_heatmap": {"w": GRID_W, "h": GRID_H, "cells": heat},
        "timeline": {"values": strip},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out}  frames={tot_frames:,} objects={tot_obj:,} poses={tot_pose:,}")
    for r in sub_rows.values():
        print(f"  {r['name']:<11} objects={r['objects']:,} confident={r['confident']:,} poses={r['poses']:,} setups={len(r['setups'])}")


if __name__ == "__main__":
    main()
