#!/usr/bin/env python
"""Verify the yaw augmentation keeps GT consistent with the points.

Same protocol that caught the flip bug: quantities that must be invariant under
a rigid rotation are (a) |origin - part centroid|, (b) angle between the axis
and the origin-to-centroid direction, (c) |axis_z|. Also checks that the
azimuth actually moved (a no-op transform would pass invariance vacuously).
"""
import sys

import numpy as np

sys.path.insert(0, "/data/challenge/USDNet")
import datasets.semseg as S

KW = dict(dataset_name="articulate3d",
          data_dir="/data/challenge/USDNet/data/processed/articulate3d_challenge_mov",
          label_db_filepath="/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/label_database.yaml",
          color_mean_std="/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/color_mean_std.yaml",
          mode="train", num_labels=3, add_raw_coordinates=True, add_normals=False,
          add_instance=True, cropping=False, point_per_cut=0, instance_oversampling=0.0,
          load_articulation=True, use_coarse_to_fine=False)

ds0 = S.SemanticSegmentationArticulationDataset(random_flip=False, random_rot_z=0.0, **KW)
ds1 = S.SemanticSegmentationArticulationDataset(random_flip=False, random_rot_z=np.pi, **KW)
idx = next(i for i, e in enumerate(ds0.data) if "08bbbdcc3d" in str(e.get("filepath", "")))


def geo(ds, seed):
    np.random.seed(seed)
    import random as _r
    _r.seed(seed)
    out = ds[idx]
    coords, labels, arti = np.asarray(out[0]), np.asarray(out[2]), out[8]
    res = {}
    for iid, g in arti.items():
        m = labels[:, 1] == int(iid)
        if m.sum() < 3:
            continue
        c = coords[m].mean(0)
        o = np.asarray(g["origin"], float)
        a = np.asarray(g["axis"], float)
        a = a / max(np.linalg.norm(a), 1e-9)
        v = o - c
        d = float(np.linalg.norm(v))
        azim = float(np.degrees(np.arctan2(a[1], a[0])))
        res[int(iid)] = (d, abs(float(np.dot(a, v / max(d, 1e-9)))), abs(float(a[2])), azim)
    return res


g0 = geo(ds0, 0)
g1 = geo(ds1, 1)
assert g0 and g1, "no instances extracted"
bad = 0
moved = 0
print(f"{'inst':>5} {'dist':>16} {'|cos(a,dir)|':>16} {'|axis_z|':>14}   azim(no/rot)")
for k in sorted(set(g0) & set(g1)):
    d0, c0, z0, az0 = g0[k]
    d1, c1, z1, az1 = g1[k]
    ok = abs(d0 - d1) < 1e-3 and abs(c0 - c1) < 1e-3 and abs(z0 - z1) < 1e-4
    bad += (not ok)
    if abs(((az1 - az0 + 180) % 360) - 180) > 1.0:
        moved += 1
    print(f"{k:5d} {d0:7.4f}/{d1:7.4f} {c0:7.4f}/{c1:7.4f} {z0:6.4f}/{z1:6.4f}   "
          f"{az0:7.1f}/{az1:7.1f}  {'OK' if ok else 'MISMATCH'}")
print(f"\n{len(g0)} instances | invariance violations: {bad} | azimuth actually changed: {moved}")
