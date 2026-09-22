#!/usr/bin/env python
"""How much of the origin gap is reachable without retraining?

Origin is the weak factor: 82.4% retention on test, and the arithmetic says a
perfect origin alone would put the ranking metric at exactly 0.410. But
"perfect" is not the question -- the question is which part of the 15% gap is
reachable by better post-processing and which needs better masks.

The deployed rule snaps a rotation origin to one of the two robust endpoints of
its own mask, measured perpendicular to the axis, choosing the side opposite the
predicted handle. Three ceilings, in increasing order of what they assume:

  1. deployed          what we score now
  2. best endpoint     oracle over the two candidates -- isolates SIDE CHOICE,
                       which is pure post-processing and costs nothing to fix
  3. best mask point   oracle over every point in the predicted mask -- the
                       ceiling of the mask itself; anything above this needs
                       better segmentation, i.e. retraining

The gap 1->2 is free. The gap 2->3 is a rule redesign. Anything beyond 3 is the
expensive path.
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


def dist_to_line(p, gt_origin, axis):
    a = unit(np.asarray(axis, float))
    d = np.asarray(p, float) - np.asarray(gt_origin, float)
    return np.linalg.norm(d - np.outer(d @ a, a), axis=-1) if d.ndim > 1 else \
        float(np.linalg.norm(d - np.dot(d, a) * a))


preds = pickle.load(open(PKL, "rb"))
dep, best_end, best_pt = [], [], []
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

        dep.append(max(dist_to_line(org[qi], gorg, gax),
                       dist_to_line(org[qi], gorg, ax[qi])))

        # the two endpoint candidates the deployed rule chooses between
        u = unit(ax[qi])
        c = pts.mean(0)
        X = pts - c
        X = X - np.outer(X @ u, u)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        d = unit(vt[0] - (vt[0] @ u) * u)
        t = X @ d
        lo, hi = np.percentile(t, [QLO, QHI])
        cand = np.stack([c + lo * d, c + hi * d])
        e = [max(dist_to_line(p, gorg, gax), dist_to_line(p, gorg, ax[qi]))
             for p in cand]
        best_end.append(min(e))

        # ceiling of the mask itself: the best point it contains
        s = pts if len(pts) <= 6000 else pts[np.random.RandomState(0).choice(len(pts), 6000, False)]
        d1 = dist_to_line(s, gorg, gax)
        d2 = dist_to_line(s, gorg, ax[qi])
        best_pt.append(float(np.maximum(d1, d2).min()))

for name, v in (("deployed", dep), ("best endpoint (side oracle)", best_end),
                ("best point in mask (mask ceiling)", best_pt)):
    v = np.array(v)
    print(f"{name:36s} pass {100*np.mean(v < THRESH):5.1f}%   median {np.median(v):.3f} m")

d, b, p = np.array(dep), np.array(best_end), np.array(best_pt)
print(f"\nn = {len(d)}")
print(f"free (fix side choice only)   : +{100*(np.mean(b<THRESH)-np.mean(d<THRESH)):.1f} pts")
print(f"rule redesign (endpoint->any) : +{100*(np.mean(p<THRESH)-np.mean(b<THRESH)):.1f} pts")
print(f"needs better masks            : {100*(1-np.mean(p<THRESH)):.1f}% unreachable")
