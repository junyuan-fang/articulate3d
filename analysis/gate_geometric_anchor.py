#!/usr/bin/env python
"""Would a GEOMETRIC anchor reach the lateral hinges the free-axis anchor cannot?

The axis head picks between gravity and `free` flattened into the ground plane.
For rotations `free` has collapsed onto gravity, so flattening it yields an
azimuth built from numerical residue -- and mask3d.py:660 falls back to `free`
itself when that residue is under 1e-4, making both anchors vertical. If that is
what happens, no selector can recover a lateral hinge and fixing the softmax
wall buys nothing.

But the direction a bottom-hinged door turns about is not mysterious: it is the
horizontal direction lying IN the door panel, along its bottom edge. We already
compute exactly that vector for the sliding-leaf rule. So ask whether replacing
the free-derived anchor with a geometry-derived one puts the lateral hinges
within reach:

    for each matched lateral hinge, angle(panel in-plane horizontal, GT axis)

This needs no new inference and no training -- only the deployed predictions and
the scene points. It says whether the information exists in the geometry at all,
which is the precondition for any anchor redesign being worth training.
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
Z = np.array([0.0, 0.0, 1.0])


def ang(a, b):
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


def panel_dirs(pts):
    """Panel normal (horizontal) and the in-plane horizontal direction."""
    c = pts - pts.mean(0)
    s = c if len(c) <= 4000 else c[np.random.RandomState(0).choice(len(c), 4000, False)]
    _, sv, vt = np.linalg.svd(s, full_matrices=False)
    n = vt[2].copy()
    n[2] = 0.0
    nn = np.linalg.norm(n)
    if nn < 1e-6:
        return None, None, float(sv[2] / max(sv[1], 1e-9))
    n = n / nn
    return n, np.array([-n[1], n[0], 0.0]), float(sv[2] / max(sv[1], 1e-9))


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
                gt[int(k)] = np.asarray(g["axis"], float)
    if not gt:
        continue
    gmasks = {i: inst == i for i in gt}
    pm = np.asarray(preds[scene]["pred_masks"]) > MASK_TH
    sc = np.asarray(preds[scene]["pred_scores"]).ravel()
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
        nrm, inp, pl = panel_dirs(pts)
        if inp is None:
            continue
        gax = gt[bid]
        rows.append(dict(scene=scene,
                         lateral=abs(gax[2] / max(np.linalg.norm(gax), 1e-9)) < 0.5,
                         a_z=ang(Z, gax), a_inp=ang(inp, gax), a_nrm=ang(nrm, gax),
                         pl=pl, score=float(sc[qi])))

lat = [r for r in rows if r["lateral"]]
ver = [r for r in rows if not r["lateral"]]
print(f"matched rotations {len(rows)}   lateral {len(lat)}   vertical {len(ver)}\n")

print("LATERAL hinges vs a geometry-derived anchor:")
print(f"{'scene':12s} {'gravity':>8s} {'panel in-plane':>15s} {'panel normal':>13s} {'plan':>6s}")
for r in sorted(lat, key=lambda x: x["a_inp"]):
    print(f"{r['scene']:12s} {r['a_z']:8.1f} {r['a_inp']:15.1f} {r['a_nrm']:13.1f} {r['pl']:6.3f}")

if lat:
    v = np.array([r["a_inp"] for r in lat])
    print(f"\npanel in-plane on laterals: {int((v < 15).sum())}/{len(lat)} within 15 deg, "
          f"median {np.median(v):.1f} deg")

orc = np.array([min(r["a_z"], r["a_inp"]) for r in rows])
cur = np.array([r["a_z"] for r in rows])
print(f"\nall matched rotations:")
print(f"  forced vertical (deployed)          : {100*np.mean(cur < 15):5.1f}% pass")
print(f"  oracle over {{gravity, panel in-plane}}: {100*np.mean(orc < 15):5.1f}% pass")
print(f"  -> ceiling if a perfect selector chose between those two: "
      f"{100*(np.mean(orc < 15) - np.mean(cur < 15)):+.1f} points")
