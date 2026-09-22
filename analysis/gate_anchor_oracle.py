#!/usr/bin/env python
"""Feasibility gate for the Riemannian direction: can the anchor set even
represent a lateral hinge?

Reading models/mask3d.py:654-684, the axis head does not regress freely. It
picks between exactly two candidates:

    free    = normalize(mask-mean of per-point axis)      (detached)
    anchor0 = gravity  [0,0,1]
    anchor1 = free flattened into the ground plane        (keeps free's azimuth)
    prior   = axis_prior_temp * |cos(anchor, free)|
    pick    = argmax(prior + cls_head(feat))              (straight-through)

We have already shown the selector cannot move: behind the temperature-10 prior
its softmax gradients are O(1e-5), which is why the anchor classifier failed.
The Riemannian proposal replaces that discrete selector with continuous
regression in the tangent space, removing the wall by construction.

But that is only worth doing if a working selector would actually help. Anchor 1
is horizontal, so a lateral hinge is representable in principle -- provided its
AZIMUTH is right. This script tests exactly that, and nothing else:

    for each matched lateral hinge, how far is anchor1 from the GT axis?

If anchor1 is close, the anchors are fine and only the selector is broken, so
the reparameterisation has something to win. If anchor1 is also ~90 deg off,
then no selector however perfect can recover these, and the direction is dead
before any training is spent on it.

Run against the AXIS_BIN=false predictions, whose pred_axises IS `free`.
"""
import os
import pickle
import sys

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
FREE = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/challenge/USDNet/results/ev_freeaxis/debug/val_preds/preds.pkl"
MASK_TH = 1e-3
ROT = 1
Z = np.array([0.0, 0.0, 1.0])


def ang(a, b):
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


preds = pickle.load(open(FREE, "rb"))
rows = []
for scene in sorted(preds):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
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
        gax = gt[bid]
        free = ax[qi] / max(np.linalg.norm(ax[qi]), 1e-9)
        flat = free - free[2] * Z
        anchor1 = flat / np.linalg.norm(flat) if np.linalg.norm(flat) > 1e-4 else free
        rows.append(dict(scene=scene, lateral=abs(gax[2] / max(np.linalg.norm(gax), 1e-9)) < 0.5,
                         a0=ang(Z, gax), a1=ang(anchor1, gax), score=float(sc[qi])))

lat = [r for r in rows if r["lateral"]]
ver = [r for r in rows if not r["lateral"]]
print(f"matched rotations {len(rows)}   lateral {len(lat)}   vertical {len(ver)}\n")

print("LATERAL hinges -- the ones a working selector would have to save:")
print(f"{'scene':12s} {'anchor0(z)':>11s} {'anchor1(free azim)':>19s} {'oracle':>7s}")
for r in sorted(lat, key=lambda x: x["a1"]):
    print(f"{r['scene']:12s} {r['a0']:11.1f} {r['a1']:19.1f} {min(r['a0'], r['a1']):7.1f}")

if lat:
    orc = np.array([min(r["a0"], r["a1"]) for r in lat])
    print(f"\nanchor oracle on laterals: {int((orc < 15).sum())}/{len(lat)} "
          f"would pass, median {np.median(orc):.1f} deg")

orc_all = np.array([min(r["a0"], r["a1"]) for r in rows])
cur = np.array([r["a0"] for r in rows])          # what forcing vertical gives
print(f"\nall matched rotations:")
print(f"  forced vertical (deployed): {100*np.mean(cur < 15):.1f}% pass")
print(f"  perfect anchor selector   : {100*np.mean(orc_all < 15):.1f}% pass")
print(f"  -> headroom from fixing the selector alone: "
      f"{100*(np.mean(orc_all < 15) - np.mean(cur < 15)):+.1f} points")
