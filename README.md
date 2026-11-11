# KITchen

KITchen is a real-world RGB-D dataset and benchmark for 6D object pose estimation in kitchen environments. It was recorded with the head camera of the ARMAR humanoid robot in two kitchen rooms, with objects on shelves and counters, in sinks, dishwashers and more.

- Website: https://kitchen-dataset.github.io/KITchen/
- Explore the data: https://kitchen-dataset.github.io/KITchen/explore.html
- Benchmark: https://kitchen-dataset.github.io/KITchen/benchmark.html
- Paper (Humanoids 2024): https://ieeexplore.ieee.org/document/10769947

Questions about the dataset: open an issue in this repository or write to kitchendataset@gmail.com.

## Download

The dataset is in this Google Drive folder:
https://drive.google.com/drive/folders/1K38tgvQN_yl9MOdF1MlKL9oJVHQyFkyC

It comes in four parts. For each part N (1 to 4) there is an RGB archive (`KITchen_rgb_<N>`), a labels archive (`KITchen_labels_<N>`) and depth archives, plus the 3D models of the objects.

Google limits large anonymous downloads. The labels archives are small, so you can fetch those first. For the full dataset, sign in to Google in the browser, or download with [rclone](https://rclone.org/drive/) or [gdown](https://github.com/wkentaro/gdown) under your own Google account.

If you only need the benchmark, its frozen splits and ground truth (COCO and BOP formats) are one zip on GitHub, with no Drive limits:
https://github.com/KITchen-Dataset/KITchenBenchmark/releases/tag/v1.0

## What's inside

- **RGB**: 1920×1080 images from the robot's head camera
- **Depth**: raw and completed, aligned to RGB
- **2D boxes**: YOLO format, 112 class ids (111 objects plus "other")
- **Instance masks**: one binary PNG per object, plus polygons
- **6D poses**: BOP-style `scene_gt.json` per part
- **3D models**: 111 textured meshes (`obj_XXXXXX.ply`)
- **Confidence**: `ignore.json` lists the ~4% of 2D labels that are unconfirmed (kept, but outside the confident set)

## File layout

Every image has a six-digit stem, such as `000000`. All files of that image use the same stem.

```
KITchen_rgb_<N>/images/<stem>.png    RGB image, 1920x1080
KITchen_labels_<N>/
  classes.txt               class names; line k (from 0) is class k
  labels/<stem>.txt         boxes: class cx cy w h, normalised to 0..1
  seg_labels/<stem>.txt     instance polygons, same row order as labels/
  masks/<stem>_obj0000.png  binary mask of object 0, then obj0001, ...
  ignore.json               {stem: [row indices of unconfirmed labels]}
  scene_gt.json             6D poses, BOP style
depth archives              <stem>.png, aligned to RGB
```

Object *i* of an image is line *i* in `labels/<stem>.txt` and `seg_labels/<stem>.txt`, and mask `masks/<stem>_obj<i>.png` (four digits).

## 6D poses

`scene_gt.json` is keyed by the image stem. Each entry is one object pose:

```json
{"000000": [{"obj_id": 5, "cam_R_m2c": [9 values, row-major], "cam_t_m2c": [x, y, z]}]}
```

- `obj_id` is the class id from `classes.txt`, and the mesh is `obj_<obj_id, 6 digits>.ply` (for example `obj_000005.ply`). Class 0 ("other") has no mesh.
- `cam_R_m2c` rotates model coordinates into the camera frame (3x3, row-major).
- `cam_t_m2c` is the translation in millimetres.
- Poses are listed only where they were accepted, so some labelled objects have no entry.

Colour camera intrinsics (1920x1080):

```
fx = 918.7635   fy = 918.3555   cx = 957.855   cy = 552.21
```

```python
import json
import numpy as np
from pathlib import Path

gt = json.loads(Path("KITchen_labels_1/scene_gt.json").read_text())

K = np.array([[918.7635, 0.0, 957.855],
              [0.0, 918.3555, 552.21],
              [0.0, 0.0, 1.0]])

for p in gt.get("000000", []):
    R = np.array(p["cam_R_m2c"]).reshape(3, 3)  # model to camera
    t = np.array(p["cam_t_m2c"]) / 1000.0       # millimetres to metres
    mesh = f"obj_{p['obj_id']:06d}.ply"
    print(mesh, R, t)
```

## 2D labels

```python
import json, cv2
from pathlib import Path

root    = Path("KITchen_labels_1")
classes = root.joinpath("classes.txt").read_text().split("\n")
ignore  = json.loads(root.joinpath("ignore.json").read_text())

stem = "000000"
rows = [l.split() for l in (root / "labels" / f"{stem}.txt").read_text().splitlines() if l]
untrusted = set(ignore.get(stem, []))

for i, row in enumerate(rows):
    if i in untrusted:
        continue          # or: mask out the loss here
    cls, cx, cy, w, h = int(row[0]), *map(float, row[1:5])
    mask = cv2.imread(str(root / "masks" / f"{stem}_obj{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
    print(classes[cls], (cx, cy, w, h), mask.shape)
```

## Benchmark

The KITchen benchmark has fixed train, validation and test splits (train in the lab room, validation and test in the built-in kitchen) and an evaluation tool:

```
pip install "git+https://github.com/KITchen-Dataset/KITchenBenchmark@v1.0"
```

See https://github.com/KITchen-Dataset/KITchenBenchmark for the splits, metrics and evaluation.

## License

The KITchen dataset (images, depth, annotations, 6D poses and 3D models) is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). You may use, share and adapt it, also commercially, as long as you cite the paper below and release adapted data under the same license.

## Citation

```bibtex
@inproceedings{younes2024kitchen,
  title     = {{KITchen}: A Real-World Benchmark and Dataset for 6D Object Pose Estimation in Kitchen Environments},
  author    = {Younes, Abdelrahman and Asfour, Tamim},
  booktitle = {2024 IEEE-RAS 23rd International Conference on Humanoid Robots (Humanoids)},
  pages     = {803--810},
  year      = {2024},
  doi       = {10.1109/Humanoids58906.2024.10769947}
}
```

## This repository

This repository also holds the source of the website (`index.html`, `explore.html`, `benchmark.html`, `static/`, and the scripts in `scripts/` that build its statistics).
