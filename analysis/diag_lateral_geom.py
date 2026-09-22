#!/usr/bin/env python
"""What do the remaining lateral-hinge failures look like geometrically?

Feature probes on all five FPN levels found no signal for lateral-vs-vertical in
the frozen backbone, and three geometric rules were tried and failed. But those
were measured on an older pipeline and an older failure set, and the deployed
pipeline's rotation failures are now eight high-confidence parts at ~90 deg
worth ~+0.017 val AP50_axis -- more than the +0.0075 needed to take the axis
factor. So look again, at the actual parts.

The physical question: a door hinged about a VERTICAL axis pivots on a vertical
(left/right) edge; a lid or oven door hinged about a HORIZONTAL axis pivots on a
horizontal (top/bottom) edge. Two things might reveal which:
  * the panel's own attitude -- a lid lies flat, so its normal points up;
  * where the GT axis sits relative to the part -- along a horizontal edge, and
    at the top or the bottom of it.
Dump both for failures and passes and see whether anything separates.
"""
import os
import pickle
import sys

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
PKL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_leaf.pkl"
MASK_TH = 1e-3
ROT = 1


def ang(a, b):
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


def describe(pts):
    c = pts - pts.mean(0)
    s = c if len(c) <= 4000 else c[np.random.RandomState(0).choice(len(c), 4000, False)]
    _, sv, vt = np.linalg.svd(s, full_matrices=False)
    nz = abs(float(vt[2][2]))                 # 1 = panel lies flat, 0 = upright
    return dict(nz=nz, planarity=float(sv[2] / max(sv[1], 1e-9)),
                h=float(np.ptp(pts[:, 2])), zmin=float(pts[:, 2].min()),
                zmax=float(pts[:, 2].max()),
                w=float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))))


preds = pickle.load(open(PKL, "rb"))
rows = []
for scene in sorted(preds):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    pts_all = np.asarray(a[:, 0:3])
    inst = np.asarray(a[:, 10]).astype(int)
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            g = h[k]
            if int(np.asarray(g["sem_id"])) == ROT:
                gt[int(k)] = (np.asarray(g["axis"], float),
                              np.asarray(g["origin"], float))
    if not gt:
        continue
    gmasks = {i: inst == i for i in gt}
    pm = np.asarray(preds[scene]["pred_masks"]) > MASK_TH
    sc = np.asarray(preds[scene]["pred_scores"]).ravel()
    ax = np.asarray(preds[scene]["pred_axises"], float)
    used = set()
    for qi in np.argsort(-sc):
        m = pm[:, qi]
        if not m.any():
            continue
        best, bid = 0.0, None
        for gi, gm in gmasks.items():
            if gi in used:
                continue
            iou = (m & gm).sum() / max((m | gm).sum(), 1)
            if iou > best:
                best, bid = iou, gi
        if best <= 0.5 or bid is None:
            continue
        used.add(bid)
        gax, gorg = gt[bid]
        pts = pts_all[m]
        if len(pts) < 10:
            continue
        d = describe(pts)
        d.update(scene=scene, err=ang(ax[qi], gax), score=float(sc[qi]),
                 gt_vert=abs(float(gax[2] / max(np.linalg.norm(gax), 1e-9))))
        # where does the GT axis sit inside the part, vertically? 0 = bottom edge,
        # 1 = top edge. Only meaningful for a horizontal axis.
        span = max(d["zmax"] - d["zmin"], 1e-6)
        d["axis_z_frac"] = (float(gorg[2]) - d["zmin"]) / span
        rows.append(d)

fail = [r for r in rows if r["err"] >= 15]
ok = [r for r in rows if r["err"] < 15]
lat = [r for r in rows if r["gt_vert"] < 0.5]        # GT axis is horizontal
print(f"matched rotations {len(rows)}   failures {len(fail)}   "
      f"GT-lateral (horizontal axis) {len(lat)}\n")

print("GT-lateral parts (what we must detect):")
print(f"{'scene':12s} {'err':>6s} {'score':>6s} {'|nz|':>6s} {'plan':>6s} "
      f"{'h':>5s} {'w':>5s} {'zmin':>6s} {'axisZ':>6s}")
for r in sorted(lat, key=lambda x: -x["err"]):
    print(f"{r['scene']:12s} {r['err']:6.1f} {r['score']:6.2f} {r['nz']:6.2f} "
          f"{r['planarity']:6.3f} {r['h']:5.2f} {r['w']:5.2f} {r['zmin']:6.2f} "
          f"{r['axis_z_frac']:6.2f}")

vert = [r for r in rows if r["gt_vert"] >= 0.5]
print(f"\nseparation check -- GT-lateral vs GT-vertical:")
for k in ("nz", "planarity", "h", "w", "zmin"):
    p = np.array([r[k] for r in lat], float)
    q = np.array([r[k] for r in vert], float)
    print(f"  {k:10s} lateral med {np.median(p):7.3f} [{p.min():6.2f} {p.max():6.2f}]"
          f"   vertical med {np.median(q):7.3f} [{q.min():6.2f} {q.max():6.2f}]")
