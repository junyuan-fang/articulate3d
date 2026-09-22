#!/usr/bin/env python
"""Export scene 280b83fcf3's point cloud from USDNet's preprocessed .npy.

ScanNet++ has withdrawn every scans/* asset for this scene (mesh, point cloud
and semantic mesh all 404, while its dslr/ and iphone/ data still serve 200),
and it no longer appears in any ScanNet++ split file. version=v2 is refused
outright (406, "download script out of date") for every file, and a v2 mesh
would be useless here anyway: Articulate3D's _parts.json indexes mesh *faces*,
so a re-decimated v2 mesh would misalign every annotation.

What we can recover is the point cloud USDNet actually trains on, which is
baked into the preprocessed data. This is NOT the original mesh:
  - points, no faces
  - voxel-downsampled at preprocessing time
  - already in the preprocessed coordinate frame
It is written outside the scannetpp_data tree, under a name that says what it
is, so nothing mistakes it for the authentic ScanNet++ asset.

Columns in the .npy: 0-2 xyz, 3-5 rgb, 6-8 normal, 9 semantic, 10/12 instance.
"""
from pathlib import Path

import numpy as np
import open3d as o3d

SRC = Path("/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/train/280b83fcf3.npy")
OUT_DIR = Path("/data/challenge/derived/280b83fcf3")
OUT = OUT_DIR / "280b83fcf3_pointcloud_from_preprocessed.ply"

a = np.load(SRC)
OUT_DIR.mkdir(parents=True, exist_ok=True)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(a[:, 0:3].astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(a[:, 3:6].astype(np.float64) / 255.0)
pcd.normals = o3d.utility.Vector3dVector(a[:, 6:9].astype(np.float64))
o3d.io.write_point_cloud(str(OUT), pcd, write_ascii=False)

back = o3d.io.read_point_cloud(str(OUT))
print(f"wrote {OUT}")
print(f"  points in npy : {len(a)}")
print(f"  points in ply : {len(back.points)}")
print(f"  colors/normals: {back.has_colors()} / {back.has_normals()}")
print(f"  size          : {OUT.stat().st_size / 1e6:.1f} MB")

(OUT_DIR / "README.txt").write_text(
    "280b83fcf3 -- point cloud recovered from USDNet's preprocessed data.\n\n"
    "ScanNet++ no longer serves any scans/* asset for this scene (all 404) and\n"
    "dropped it from every split file, so the original mesh_aligned_0.05.ply is\n"
    "not obtainable. This file is NOT that mesh: it has no faces and is\n"
    "voxel-downsampled. Do not feed it to the ScanNet++ preprocessing scripts as\n"
    "if it were the raw asset.\n\n"
    "Nothing in the USDNet pipeline needs it -- training and inference read\n"
    "data/processed/.../train/280b83fcf3.npy, which is present, so the train\n"
    "split is a complete 195 scenes. This export exists only for inspection and\n"
    "visualisation.\n"
)
