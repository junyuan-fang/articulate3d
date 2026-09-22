#!/usr/bin/env python
"""Ceiling of a three-anchor geometric candidate set, over ALL matched instances.

Earlier gates looked only at rotations, and docs/07 4.7 proposed replacing the
free-derived anchor with the panel in-plane direction. That is incomplete: two
anchors cannot cover the three things a flat part actually does.

    vertical hinge            axis = gravity
    drawer                    axis = panel normal      (through the face)
    sliding leaf / bottom hinge  axis = panel in-plane (along the face)

So the candidate set should be {gravity, panel normal, panel in-plane}, and the
head's job reduces to a 3-way choice over quantities computed from the mask.

This measures the ceiling of that set before any training is written: what
fraction of matched instances would pass the 15 deg axis test if the choice were
always right, split by articulation type, and how often each anchor wins. If the
ceiling is not clearly above what the deployed geometry stack already achieves,
the redesign is not worth training.
"""
import os
import pickle
import sys
from collections import Counter

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
PKL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_leaf.pkl"
MASK_TH = 1e-3
ROT, TRANS = 1, 2
Z = np.array([0.0, 0.0, 1.0])


def ang(a, b):
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


def anchors_for(pts):
    c = pts - pts.mean(0)
    s = c if len(c) <= 4000 else c[np.random.RandomState(0).choice(len(c), 4000, False)]
    _, _, vt = np.linalg.svd(s, full_matrices=False)
    n = vt[2].copy(); n[2] = 0.0
    nn = np.linalg.norm(n)
    if nn < 1e-6:
        return None
    n = n / nn
    return [Z, n, np.array([-n[1], n[0], 0.0])]


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
            gt[int(k)] = (int(np.asarray(g["sem_id"])), np.asarray(g["axis"], float))
    if not gt:
        continue
    gmasks = {i: inst == i for i in gt}
    pm = np.asarray(preds[scene]["pred_masks"]) > MASK_TH
    sc = np.asarray(preds[scene]["pred_scores"]).ravel()
    cur = np.asarray(preds[scene]["pred_axises"], float)
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
        pts = pts_all[m]
        if len(pts) < 10:
            continue
        anc = anchors_for(pts)
        if anc is None:
            continue
        sem, gax = gt[bid]
        errs = [ang(x, gax) for x in anc]
        rows.append(dict(sem=sem, errs=errs, deployed=ang(cur[qi], gax)))

names = ["gravity", "panel normal", "panel in-plane"]
for label, sem in (("rotation", ROT), ("translation", TRANS), ("ALL", None)):
    g = [r for r in rows if sem is None or r["sem"] == sem]
    if not g:
        continue
    dep = np.array([r["deployed"] for r in g])
    orc = np.array([min(r["errs"]) for r in g])
    win = Counter(names[int(np.argmin(r["errs"]))] for r in g if min(r["errs"]) < 15)
    print(f"{label:12s} n={len(g):3d}   deployed {100*np.mean(dep<15):5.1f}%   "
          f"3-anchor oracle {100*np.mean(orc<15):5.1f}%   "
          f"(+{100*(np.mean(orc<15)-np.mean(dep<15)):.1f} pts)")
    print(f"{'':12s} winning anchor among passes: "
          + ", ".join(f"{k} {v}" for k, v in win.most_common()))

print("\nHow often is the deployed answer already the best anchor?")
same = sum(1 for r in rows if abs(min(r["errs"]) - r["deployed"]) < 1.0)
print(f"  {same}/{len(rows)}")
