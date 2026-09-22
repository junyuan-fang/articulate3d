#!/usr/bin/env python
"""Which direction is the origin error in?

gate_origin_oracle.py showed the mask ceiling for origin is 100% -- the right
point is always inside the predicted mask, median 6 mm from the GT axis line --
so the 15% shortfall is entirely a rule-design problem, not a segmentation one.

The deployed rule places the origin at `c + t*d`, where c is the mask centroid,
d is the widest direction perpendicular to the axis, and t is a 10/90 quantile.
Distance to the GT axis line ignores displacement ALONG the axis, so the error
lives in the 2-D plane perpendicular to it, and splits into two components:

    along d   the left/right placement -- wrong side, or wrong extent
    along e   the remaining direction (d x axis): for a door, its depth. The
              hinge sits at the back face where the door meets its frame, but
              the centroid sits mid-thickness, so this component is a
              systematic offset the current rule cannot express at all.

If the error is mostly along e, the fix is to move the origin to the part's
extreme in depth rather than its centroid -- a small change. If it is along d,
the fix is in the side/quantile choice.
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
THRESH = 0.25
QLO, QHI = 10.0, 90.0


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


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
    org = np.asarray(preds[scene]["pred_origins"], float)
    used = set()
    for qi in np.argsort(-sc):
        m = pm[:, qi]
        if not m.any():
            continue
        bi, bid = 0.0, None
        for gi, gm in gmasks.items():
            if gi in used:
                continue
            iou = (m & gm).sum() / max((m | gm).sum(), 1)
            if iou > bi:
                bi, bid = iou, gi
        if bi <= 0.5 or bid is None:
            continue
        used.add(bid)
        gax, gorg = gt[bid]
        pts = pts_all[m]
        if len(pts) < 3:
            continue

        u = unit(ax[qi])
        c = pts.mean(0)
        X = pts - c
        Xp = X - np.outer(X @ u, u)
        _, _, vt = np.linalg.svd(Xp, full_matrices=False)
        d = unit(vt[0] - (vt[0] @ u) * u)
        e = unit(np.cross(u, d))

        err = np.asarray(org[qi], float) - np.asarray(gorg, float)
        err_perp = err - np.dot(err, u) * u          # what the metric sees
        rows.append(dict(total=float(np.linalg.norm(err_perp)),
                         along_d=abs(float(np.dot(err, d))),
                         along_e=abs(float(np.dot(err, e))),
                         depth_extent=float(np.ptp(X @ e)),
                         width_extent=float(np.ptp(X @ d))))

tot = np.array([r["total"] for r in rows])
dd = np.array([r["along_d"] for r in rows])
ee = np.array([r["along_e"] for r in rows])
fail = tot >= THRESH
print(f"matched rotations {len(rows)}   origin failures {int(fail.sum())}\n")
for name, sel in (("all", np.ones(len(rows), bool)), ("failures only", fail)):
    print(f"{name}:")
    print(f"  error along d (width  direction): median {np.median(dd[sel]):.3f} m")
    print(f"  error along e (depth  direction): median {np.median(ee[sel]):.3f} m")
    print(f"  total perpendicular             : median {np.median(tot[sel]):.3f} m")

print(f"\nfailures where d dominates (d > 2e): "
      f"{int((fail & (dd > 2*ee)).sum())}/{int(fail.sum())}")
print(f"failures where e dominates (e > 2d): "
      f"{int((fail & (ee > 2*dd)).sum())}/{int(fail.sum())}")
print(f"failures where both matter        : "
      f"{int((fail & (dd <= 2*ee) & (ee <= 2*dd)).sum())}/{int(fail.sum())}")

de = np.array([r["depth_extent"] for r in rows])
print(f"\npart depth extent: median {np.median(de):.3f} m "
      f"(if the hinge is at a face and we sit mid-depth, expect ~half of this)")
