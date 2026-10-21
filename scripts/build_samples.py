#!/usr/bin/env python3
"""Export a handful of frames for the website's interactive viewer.

    python scripts/build_samples.py --release /path/to/kat_release \
        --rgb-root /path/to/DATA --depth-root /path/to/DATA \
        --split-file splits/setups.csv --models static/models 000500 002000 ...

For every stem it writes into static/samples/:
    <stem>.jpg         RGB, 1920x1080
    <stem>_t.jpg       480 px thumbnail
    <stem>_depth.jpg   completed depth, colour-mapped (near = warm)
    <stem>_pose.png    rendered 6D poses (transparent overlay)
and writes all exported frames' annotations (with their benchmark split) to
data/explorer_frames.json, replacing that file.

IMPORTANT: only export frames that show no people, or face-blurred frames.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from build_stats import read_split_file

FX, FY, CX, CY = 918.7635, 918.3555, 957.855, 552.21      # Azure Kinect colour camera @1920x1080
W, H = 1920, 1080
REPO = Path(__file__).resolve().parent.parent


def seq_of(stem: str) -> int:
    return int(stem) // 25000 + 1


def find(root: Path, seq: int, kind: str, stem: str) -> Path:
    cands = [root / f"KITchen_{kind}_{seq}" / "images" / f"{stem}.png",
             root / f"KITchen_{kind}_{seq}" / "depth_completed" / f"{stem}.png",
             root / f"KITchen_{kind}_{seq}" / f"{stem}.png"]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"{kind} for {stem}: tried {cands}")


def decode_depth(img: np.ndarray, order: str = "g_hi_r_lo") -> np.ndarray:
    """RG16 depth (RGB array) in mm.

    KITchen raw and completed depth store the high byte in G and the low byte in
    R. Per-frame guessing is unreliable on smooth completed depth (both orders can
    look equally smooth), so the order is fixed; "auto" picks the order with fewer
    neighbour jumps above 200 mm, for other datasets.
    """
    r, g = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32)
    cands = {"g_hi_r_lo": g * 256 + r, "r_hi_g_lo": r * 256 + g}
    if order != "auto":
        return cands[order]

    def jumps(d):
        v = d > 0
        return float((np.abs(np.diff(d, axis=1))[v[:, 1:] & v[:, :-1]] > 200).mean())
    return min(cands.values(), key=jumps)


def export_depth(depth_png: Path, out_jpg: Path, order: str = "g_hi_r_lo") -> tuple[int, int]:
    """Colour-map completed depth (near = warm) and return its 1st-99th percentile range in mm."""
    d = decode_depth(np.array(Image.open(depth_png).convert("RGB")), order)
    valid = d > 0
    lo, hi = np.percentile(d[valid], [1, 99]) if valid.any() else (0, 1)
    col = turbo(1.0 - (d - lo) / max(1.0, hi - lo)); col[~valid] = 0
    Image.fromarray(col).save(out_jpg, quality=82, optimize=True)
    return int(lo), int(hi)


def turbo(x: np.ndarray) -> np.ndarray:
    """Google's Turbo colormap (polynomial approximation), x in [0,1] -> uint8 RGB."""
    x = np.clip(x, 0, 1)[..., None]
    kr = np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943])
    kg = np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604])
    kb = np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973])
    p = np.concatenate([x ** i for i in range(6)], axis=-1)
    return (np.stack([p @ kr, p @ kg, p @ kb], -1).clip(0, 1) * 255).astype(np.uint8)


def palette(i: int) -> tuple[int, int, int]:
    h = (i * 0.61803398875) % 1.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, 0.72, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def load_points(models: Path, oid: int, cache: dict):
    if oid not in cache:
        import trimesh
        p = models / f"obj_{oid:06d}.glb"
        if not p.exists():
            cache[oid] = None
        else:
            m = trimesh.load(p, force="mesh")
            pts, fi = trimesh.sample.sample_surface(m, 80000, seed=0)
            cache[oid] = (pts, m.face_normals[fi])
    return cache[oid]


def render_poses(poses, models: Path, cache: dict, scale=0.5):
    w, h = int(W * scale), int(H * scale)
    zbuf = np.full((h, w), np.inf, np.float32)
    rgb = np.zeros((h, w, 3), np.float32)
    idb = np.full((h, w), -1, np.int32)
    light = np.array([0.3, -0.5, -0.8]); light /= np.linalg.norm(light)
    for k, p in enumerate(poses):
        pn = load_points(models, p["obj_id"], cache)
        if pn is None:
            continue
        pts, nrm = pn
        R = np.array(p["R"]).reshape(3, 3); t = np.array(p["t"])
        X = pts @ R.T + t; N = nrm @ R.T
        z = X[:, 2]; ok = z > 0.05
        u = (FX * X[ok, 0] / z[ok] + CX) * scale; v = (FY * X[ok, 1] / z[ok] + CY) * scale
        shade = 0.62 + 0.38 * np.clip(-(N[ok] @ light), 0, 1)
        col = np.array(palette(p["obj_id"])) / 255.0
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                ui = np.round(u).astype(int) + du; vi = np.round(v).astype(int) + dv
                m = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
                ui, vi, zz, ss = ui[m], vi[m], z[ok][m], shade[m]
                order = np.argsort(-zz)                         # far first, near overwrites
                ui, vi, zz, ss = ui[order], vi[order], zz[order], ss[order]
                closer = zz < zbuf[vi, ui]
                ui, vi, zz, ss = ui[closer], vi[closer], zz[closer], ss[closer]
                zbuf[vi, ui] = zz; rgb[vi, ui] = col * ss[:, None]; idb[vi, ui] = k
    alpha = (idb >= 0).astype(np.float32)
    # Outline where the object id changes.
    edge = np.zeros_like(alpha, bool)
    edge[:, 1:] |= idb[:, 1:] != idb[:, :-1]; edge[1:, :] |= idb[1:, :] != idb[:-1, :]
    out = np.dstack([rgb * 255, alpha * 170]).astype(np.uint8)
    out[edge & (alpha > 0) | (edge & np.roll(alpha > 0, 1, 1))] = [255, 255, 255, 255]
    return Image.fromarray(out, "RGBA")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--release", type=Path, required=True, help="folder with KITchen_labels_<N>/")
    ap.add_argument("--rgb-root", type=Path, required=True)
    ap.add_argument("--depth-root", type=Path, required=True)
    ap.add_argument("--split-file", type=Path, required=True, help="benchmark splits/setups.csv or its run list")
    ap.add_argument("--models", type=Path, default=REPO / "static" / "models")
    ap.add_argument("--out", type=Path, default=REPO / "static" / "samples")
    ap.add_argument("--json", type=Path, default=REPO / "data" / "explorer_frames.json")
    ap.add_argument("--depth-order", default="g_hi_r_lo", choices=["g_hi_r_lo", "r_hi_g_lo", "auto"],
                    help="RG16 byte order of the depth PNGs (KITchen: g_hi_r_lo)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    classes = [c for c in (a.release / "KITchen_labels_1" / "classes.txt").read_text().splitlines() if c.strip()]
    setups = read_split_file(a.split_file)
    gts, ignores, frames, cache = {}, {}, [], {}
    for stem in a.stems:
        seq = seq_of(stem); lab = a.release / f"KITchen_labels_{seq}"
        if seq not in gts:
            gts[seq] = json.loads((lab / "scene_gt.json").read_text())
            ignores[seq] = json.loads((lab / "ignore.json").read_text())
        rows = [l.split() for l in (lab / "labels" / f"{stem}.txt").read_text().splitlines() if l.strip()]
        polys = [l.split() for l in (lab / "seg_labels" / f"{stem}.txt").read_text().splitlines() if l.strip()]
        bad = set(ignores[seq].get(stem, []))

        rgb = Image.open(find(a.rgb_root, seq, "rgb", stem)).convert("RGB")
        rgb.save(a.out / f"{stem}.jpg", quality=84, optimize=True, progressive=True)
        t = rgb.copy(); t.thumbnail((480, 270), Image.LANCZOS); t.save(a.out / f"{stem}_t.jpg", quality=82)

        lo, hi = export_depth(find(a.depth_root, seq, "depth_completed", stem), a.out / f"{stem}_depth.jpg", a.depth_order)

        objs = []
        for i, r in enumerate(rows):
            c = int(r[0]); cx, cy, w, h = map(float, r[1:5])
            poly = []
            if i < len(polys) and int(polys[i][0]) == c:
                xy = np.array(list(map(float, polys[i][1:]))).reshape(-1, 2)
                step = max(1, len(xy) // 60)
                poly = [round(v, 4) for v in xy[::step].ravel()]
            objs.append({"c": c, "b": [round(cx, 4), round(cy, 4), round(w, 4), round(h, 4)],
                         "p": poly, "ok": i not in bad})

        poses = []
        for o in gts[seq].get(stem, []):
            tt = o["cam_t_m2c"]; tt = [v / 1000 for v in tt] if max(abs(v) for v in tt) > 20 else tt
            poses.append({"obj_id": o["obj_id"], "R": [round(v, 5) for v in o["cam_R_m2c"]],
                          "t": [round(v, 4) for v in tt]})
        # Attach each pose to the label row of the same class whose box centre is closest.
        used = set()
        for p in poses:
            u = (FX * p["t"][0] / p["t"][2] + CX) / W; v = (FY * p["t"][1] / p["t"][2] + CY) / H
            best = min((math.hypot(o["b"][0] - u, o["b"][1] - v), i) for i, o in enumerate(objs)
                       if o["c"] == p["obj_id"] and i not in used) if any(
                o["c"] == p["obj_id"] and i not in used for i, o in enumerate(objs)) else None
            if best:
                used.add(best[1]); objs[best[1]]["pose"] = {"R": p["R"], "t": p["t"]}
        render_poses(poses, a.models, cache).save(a.out / f"{stem}_pose.png", optimize=True)

        frames.append({"stem": stem, "seq": seq, "split": setups[stem][1] if stem in setups else "buffer", "depth_range_mm": [int(lo), int(hi)], "objects": objs})
        print(f"{stem}: {len(objs)} objects, {len(poses)} poses", flush=True)

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps({"camera": {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "w": W, "h": H},
                                  "classes": classes, "frames": frames}, separators=(",", ":")))


if __name__ == "__main__":
    main()
