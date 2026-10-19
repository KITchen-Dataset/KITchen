#!/usr/bin/env python3
"""Convert the KITchen object models (BOP-style PLY, millimetres) into web-ready GLB.

    pip install pymeshlab trimesh pillow
    python scripts/build_models.py /path/to/models --out static/models

Each model is decimated to at most --faces triangles (texture coordinates are
preserved), its texture is downscaled to --tex pixels and stored as JPEG, and
the result is written in metres as static/models/obj_XXXXXX.glb.  A small
static/models/models.json records each model's extent so the viewer can draw
3D boxes.  Source files are only read, never modified.

Six scanned models (obj 53, 54, 66, 81, 83, 95) defeat meshlab's decimation and
come out broken.  For those, bake the texture into vertex colours, merge
duplicate vertices, and simplify with gltfpack (npm i gltfpack@0.21.0):

    m = trimesh.load(ply, process=False); m.apply_scale(0.001)
    trimesh.Trimesh(m.vertices, m.faces, vertex_colors=m.visual.to_color().vertex_colors).export("vc.glb")
    gltfpack -i vc.glb -o obj_XXXXXX.glb -si <30000 / face count> -noq
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def convert(args):
    ply, out_dir, max_faces, tex_px = args
    import pymeshlab
    import trimesh
    from PIL import Image

    oid = int(re.search(r"obj_(\d+)", ply.name).group(1))
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(ply))
    m = ms.current_mesh()
    textured = bool(m.textures()) and (m.has_wedge_tex_coord() or m.has_vertex_tex_coord())
    tex_name = None
    head = ply.read_bytes()[:600].decode("latin-1")
    hit = re.search(r"TextureFile\s+(\S+)", head)
    if hit:
        tex_name = hit.group(1)
    if textured and not m.has_wedge_tex_coord():
        ms.compute_texcoord_transfer_vertex_to_wedge()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    if ms.current_mesh().face_number() > max_faces:
        if textured:
            ms.meshing_decimation_quadric_edge_collapse_with_texture(
                targetfacenum=max_faces, qualitythr=0.5, preserveboundary=True, optimalplacement=True)
        else:
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=max_faces, qualitythr=0.5, preservenormal=True, optimalplacement=True)
    if textured and ms.current_mesh().face_number() > 1.5 * max_faces:
        # Texture-preserving decimation can stall on non-manifold scans; bake the
        # texture into vertex colours instead and decimate the plain mesh.
        ms.compute_color_from_texture_per_vertex()
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=max_faces, qualitythr=0.5, preservenormal=True, optimalplacement=True)
        textured = False
    m = ms.current_mesh()
    V = m.vertex_matrix() / 1000.0                     # mm -> m
    F = m.face_matrix()
    # Web copy only: drop stray triangles far outside the object (scan artefacts).
    med = np.median(V, 0)
    vd = np.linalg.norm(V - med, axis=1)
    stray = (vd[F] > 1.8 * np.percentile(vd, 99)).any(1)
    F = F[~stray]

    if textured and tex_name and (ply.parent / tex_name).exists():
        # Split vertices per wedge so each corner keeps its own UV.
        uv = m.wedge_tex_coord_matrix().reshape(-1, 3, 2)[~stray]
        verts = V[F].reshape(-1, 3)
        faces = np.arange(len(verts)).reshape(-1, 3)
        uvs = uv.reshape(-1, 2)
        img = Image.open(ply.parent / tex_name).convert("RGB")
        img.thumbnail((tex_px, tex_px), Image.LANCZOS)
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=85); buf.seek(0)
        img = Image.open(buf)
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img, metallicFactor=0.0, roughnessFactor=0.85)
        mesh = trimesh.Trimesh(verts, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, material=mat), process=False)
        mesh.merge_vertices(merge_tex=False, merge_norm=True)
    else:
        colors = None
        if m.has_vertex_color():
            colors = (m.vertex_color_matrix() * 255).clip(0, 255).astype(np.uint8)
        if len(F) > 1.5 * max_faces:
            # Still too dense (meshlab stops early on broken topology): simplify the
            # geometry directly and carry colours over from the nearest source vertex.
            import fast_simplification
            from scipy.spatial import cKDTree
            V2, F2 = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), 1 - max_faces / len(F))
            if colors is not None:
                colors = colors[cKDTree(V).query(V2)[1]]
            V, F = V2.astype(np.float64), F2
        mesh = trimesh.Trimesh(V, F, vertex_colors=colors, process=False)
        textured = False

    out = out_dir / f"obj_{oid:06d}.glb"
    mesh.export(out)
    used = V[np.unique(F)]
    lo, hi = used.min(0), used.max(0)
    return {"id": oid, "faces": int(len(F)), "textured": textured, "bytes": out.stat().st_size, "stray_faces_dropped": int(stray.sum()),
            "min": [round(float(x), 5) for x in lo], "max": [round(float(x), 5) for x in hi],
            "diameter": round(float(np.linalg.norm(hi - lo)), 5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "static" / "models")
    ap.add_argument("--faces", type=int, default=12000)
    ap.add_argument("--tex", type=int, default=1024)
    ap.add_argument("--only", type=int, nargs="*", help="object ids to convert (default: all)")
    ap.add_argument("-j", "--jobs", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plys = sorted(p for p in a.models.glob("obj_*.ply") if re.fullmatch(r"obj_\d{6}\.ply", p.name))
    if a.only:
        plys = [p for p in plys if int(p.stem[4:]) in set(a.only)]
    meta_path = a.out / "models.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for p, res in zip(plys, ex.map(convert, [(p, a.out, a.faces, a.tex) for p in plys])):
            meta[str(res["id"])] = res
            print(f"{p.name}: {res['faces']} faces, {'tex' if res['textured'] else 'color'}, {res['bytes']/1e6:.2f} MB", flush=True)
    meta_path.write_text(json.dumps(dict(sorted(meta.items(), key=lambda kv: int(kv[0]))), separators=(",", ":")))


if __name__ == "__main__":
    main()
