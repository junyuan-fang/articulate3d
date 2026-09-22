#!/usr/bin/env python
"""Re-diagnose the CURRENT best pipeline (val 0.4306), not the baseline.

Every earlier diagnosis was run on raw USDNet output. The geometric stack has
since rewritten every axis and every rotation origin, so the残 error profile is
a different animal -- and the gap to the leader is only 0.0022, i.e. one or two
instances. This asks: among matched instances, which criterion still fails, by
how much, and how much headroom sits just past each threshold.
"""
import os
import pickle
import sys

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
FINAL = "/tmp/v4init_full.pkl"          # pca + endpoint + handle side, val 0.4306
MASK_TH = 1e-3


def perp(v, a):
    a = a / max(np.linalg.norm(a), 1e-9)
    return float(np.linalg.norm(v - (v @ a) * a))


preds = pickle.load(open(FINAL, "rb"))
rows = []
for scene in sorted(preds):
    p = preds[scene]
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    inst = np.asarray(a[:, 10]).astype(int)
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            g = h[k]
            gt[int(k)] = (np.asarray(g["axis"], float), np.asarray(g["origin"], float),
                          int(np.asarray(g["sem_id"])))
    if not gt:
        continue
    gmasks = {i: inst == i for i in gt}
    pm = np.asarray(p["pred_masks"]) > MASK_TH
    sc = np.asarray(p["pred_scores"]).ravel()
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
        gax, gor, sem = gt[bid]
        pax = np.asarray(p["pred_axises"])[qi]
        por = np.asarray(p["pred_origins"])[qi]
        ang = float(np.degrees(np.arccos(min(1.0, abs(float(
            (pax / max(np.linalg.norm(pax), 1e-9)) @ (gax / max(np.linalg.norm(gax), 1e-9))))))))
        d = por - gor
        dg = perp(d, gax)
        dp = perp(d, pax)
        rows.append((sem, ang, max(dg, dp), float(sc[qi]), int(m.sum())))

R = np.array(rows, float)
rot, tr = R[R[:, 0] == 1], R[R[:, 0] == 2]
print(f"matched {len(R)}   rotation {len(rot)}   translation {len(tr)}\n")

def rep(name, X, use_origin):
    ax_ok = X[:, 1] < 15
    or_ok = X[:, 2] < 0.25 if use_origin else np.ones(len(X), bool)
    both = ax_ok & or_ok
    print(f"{name} (n={len(X)})")
    print(f"  axis pass   {ax_ok.mean()*100:5.1f}%   origin pass {or_ok.mean()*100:5.1f}%   both {both.mean()*100:5.1f}%")
    f = ~both
    if f.any():
        print(f"  failures {f.sum()}: axis-only {(f & or_ok).sum()}  origin-only {(f & ax_ok).sum()}  "
              f"both-fail {(f & ~ax_ok & ~or_ok).sum()}" if use_origin else f"  failures {f.sum()}")
    # near-miss headroom
    for th, lbl in ((20, "axis<20"), (25, "axis<25")):
        gain = ((X[:, 1] < th) & or_ok).mean() - both.mean()
        print(f"  if {lbl}: +{gain*100:.1f}pt")
    if use_origin:
        for th, lbl in ((0.3, "origin<0.30"), (0.35, "origin<0.35")):
            gain = (ax_ok & (X[:, 2] < th)).mean() - both.mean()
            print(f"  if {lbl}: +{gain*100:.1f}pt")
    print()

rep("ROTATION", rot, True)
rep("TRANSLATION", tr, False)

print("near-miss instances (within 1.5x of a threshold):")
for sem, ang, od, s, n in rows:
    tag = None
    if sem == 1:
        if 15 <= ang < 22.5: tag = f"axis {ang:5.1f}deg"
        elif ang < 15 and 0.25 <= od < 0.375: tag = f"origin {od:.3f}m"
    else:
        if 15 <= ang < 22.5: tag = f"axis {ang:5.1f}deg"
    if tag:
        print(f"  {'rot' if sem==1 else 'tr ':3s} {tag}   score {s:.3f}  pts {n}")
