#!/usr/bin/env python
"""Where does the axis metric still lose, on the CURRENTLY deployed pipeline?

The previous breakdown was measured before the sliding-leaf flip, and the
project's own history says optimising against a stale diagnosis is the most
expensive mistake available. So re-measure: split matched instances by
articulation type, report the axis pass rate of each, and show what the
remaining failures look like.

The test gap to close is AP50_axis 0.4060 -> 0.4135.
"""
import os
import pickle
import sys

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
PKL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_leaf.pkl"
MASK_TH = 1e-3
ROT, TRANS = 1, 2


def ang(a, b):
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


preds = pickle.load(open(PKL, "rb"))
rows = []
for scene in sorted(preds):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
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
    ax = np.asarray(preds[scene]["pred_axises"], float)
    cl = np.asarray(preds[scene]["pred_classes"]).ravel()
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
        sem, gax = gt[bid]
        rows.append((scene, sem, int(cl[qi]), ang(ax[qi], gax), float(sc[qi]),
                     int(m.sum())))

print(f"file: {PKL}\nmatched instances: {len(rows)}\n")
for name, sem in (("rotation", ROT), ("translation", TRANS)):
    g = [r for r in rows if r[1] == sem]
    if not g:
        continue
    e = np.array([r[3] for r in g])
    ok = int((e < 15).sum())
    print(f"{name:12s} n={len(g):3d}   axis pass {ok}/{len(g)} = {100*ok/len(g):5.1f}%"
          f"   median {np.median(e):5.2f} deg")
    f = sorted([r for r in g if r[3] >= 15], key=lambda r: -r[3])
    print(f"  failures ({len(f)}), worst first:")
    for r in f:
        wrongtype = "  TYPE-MISPREDICTED" if r[2] != r[1] else ""
        print(f"    {r[0]}  {r[3]:5.1f} deg  score {r[4]:.2f}  pts {r[5]:5d}{wrongtype}")
    print()

# how much of the remaining loss is type confusion rather than axis regression?
bad_type = [r for r in rows if r[1] != r[2]]
print(f"type mispredictions among matched: {len(bad_type)}/{len(rows)}")
if bad_type:
    e = np.array([r[3] for r in bad_type])
    print(f"  their axis errors: median {np.median(e):.1f} deg, "
          f"{int((e >= 15).sum())}/{len(e)} fail the axis test")
