#!/usr/bin/env python3
"""Find candidate frames for the website explorer.

    python scripts/select_frames.py /path/to/kat_release candidates.json

Keeps frames where every labelled object is confident and has a 6D pose (same
class counts in labels/ and scene_gt.json, at least 12 objects), and scores each
pose by the IoU between its projected model box and the 2D label box. Output rows:
[stem, part, n_objects, min_iou, mean_iou]. Check the best candidates by eye
(pose fit, and no people in view) before exporting them with build_samples.py.
"""
import json, sys, numpy as np, trimesh
from collections import Counter
from pathlib import Path
R=sys.argv[1]; LAB=R+'/KITchen_labels_%d/labels'
FX,FY,CX,CY=918.7635,918.3555,957.855,552.21
M=Path(__file__).resolve().parent.parent/'static'/'models'; pts={}
def P(o):
    if o not in pts:
        p=M/f'obj_{o:06d}.glb'
        pts[o]=trimesh.sample.sample_surface(trimesh.load(p,force='mesh'),600,seed=0)[0] if p.exists() else None
    return pts[o]
out=[]
for s in (1,2,3,4):
    gt=json.load(open(f'{R}/KITchen_labels_{s}/scene_gt.json')); ig=json.load(open(f'{R}/KITchen_labels_{s}/ignore.json'))
    for stem,poses in gt.items():
        if ig.get(stem): continue
        f=Path(LAB%s)/f'{stem}.txt'
        if not f.exists(): continue
        rows=[l.split() for l in f.read_text().splitlines() if l.strip()]
        if len(rows)<12 or len(rows)!=len(poses): continue
        if Counter(int(r[0]) for r in rows)!=Counter(p['obj_id'] for p in poses): continue
        boxes=[(int(r[0]),*[float(v) for v in r[1:5]]) for r in rows]
        used=set(); ious=[]
        for p in poses:
            q=P(p['obj_id'])
            if q is None: ious.append(0); continue
            t=np.array(p['cam_t_m2c']); t=t/1000 if abs(t).max()>20 else t
            X=q@np.array(p['cam_R_m2c']).reshape(3,3).T+t
            if (X[:,2]<=0.05).any(): ious.append(0); continue
            u=FX*X[:,0]/X[:,2]+CX; v=FY*X[:,1]/X[:,2]+CY
            a=(u.min()/1920,v.min()/1080,u.max()/1920,v.max()/1080)
            best=(0,None)
            for i,(c,cx,cy,w,h) in enumerate(boxes):
                if c!=p['obj_id'] or i in used: continue
                b=(cx-w/2,cy-h/2,cx+w/2,cy+h/2)
                iw=max(0,min(a[2],b[2])-max(a[0],b[0])); ih=max(0,min(a[3],b[3])-max(a[1],b[1]))
                I=iw*ih; U=(a[2]-a[0])*(a[3]-a[1])+w*h-I; iou=I/U if U>0 else 0
                if iou>best[0]: best=(iou,i)
            if best[1] is not None: used.add(best[1])
            ious.append(best[0])
        out.append((stem,s,len(rows),round(min(ious),3),round(float(np.mean(ious)),3)))
    print(s,len(out),flush=True)
json.dump(out,open(sys.argv[2],'w'))
