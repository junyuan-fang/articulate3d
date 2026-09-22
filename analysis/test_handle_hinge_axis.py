#!/usr/bin/env python
"""Does the handle's position tell us the hinge ORIENTATION, not just its side?

gate_geometric_anchor.py showed the lateral hinges' axis is sitting right there
in the geometry -- the panel's in-plane horizontal direction lands within
15 deg on 5 of 7, median 3.7 deg. What is missing is a selector deciding which
rotation parts should use it, and the economics are unforgiving: a correct flip
gains one instance, a wrong flip loses one that was already passing, so the
detector needs better than 50% precision. Ten previous attempts topped out at a
coin flip.

This tries a signal none of them used. We already exploit "the hinge is opposite
the handle" to pick which END the origin sits at; the same physics constrains the
hinge's DIRECTION:

    handle displaced left/right of the part  ->  hinge on a vertical edge  -> axis vertical
    handle displaced above the part          ->  hinge on the bottom edge  -> axis horizontal

So decompose the handle-to-part offset into its vertical component and its
in-plane horizontal component, and check whether the ratio separates the seven
lateral hinges from the 173 vertical ones.
"""
import os
import pickle
import sys

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
PKL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_leaf.pkl"
INTER = "/data/challenge/USDNet/results/inference_inter_seg_val/debug/val_preds/preds.pkl"
MASK_TH = 1e-3
ROT = 1
Z = np.array([0.0, 0.0, 1.0])
HANDLE_MARGIN = 0.12


def ang(a, b):
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


def panel_inp(pts):
    c = pts - pts.mean(0)
    s = c if len(c) <= 4000 else c[np.random.RandomState(0).choice(len(c), 4000, False)]
    _, _, vt = np.linalg.svd(s, full_matrices=False)
    n = vt[2].copy(); n[2] = 0.0
    nn = np.linalg.norm(n)
    if nn < 1e-6:
        return None
    n = n / nn
    return np.array([-n[1], n[0], 0.0])


inter = pickle.load(open(INTER, "rb"))
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

    handles = []
    if scene in inter:
        im = np.asarray(inter[scene]["pred_masks"]) > MASK_TH
        isc = np.asarray(inter[scene]["pred_scores"]).ravel()
        for hi in np.where(isc > 0.5)[0]:
            sel = im[:, hi]
            if 10 <= sel.sum() < 20000:
                handles.append(pts_all[sel].mean(0))
    handles = np.array(handles) if handles else None

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
        inp = panel_inp(pts)
        if inp is None:
            continue
        gax = gt[bid]
        lateral = abs(gax[2] / max(np.linalg.norm(gax), 1e-9)) < 0.5

        dz = di = np.nan
        if handles is not None and len(handles):
            lo, hi_ = pts.min(0) - HANDLE_MARGIN, pts.max(0) + HANDLE_MARGIN
            near = handles[np.all((handles > lo) & (handles < hi_), axis=1)]
            if len(near):
                cen = pts.mean(0)
                # nearest handle to the part centre
                h = near[np.argmin(np.linalg.norm(near - cen, axis=1))]
                d = h - cen
                # normalise by the part's own extent so big and small doors compare
                ez = max(np.ptp(pts[:, 2]), 1e-6)
                ei = max(np.ptp((pts - cen) @ inp), 1e-6)
                dz = abs(float(d[2])) / ez
                di = abs(float(d @ inp)) / ei

        rows.append(dict(scene=scene, lateral=lateral, dz=dz, di=di,
                         a_inp=ang(inp, gax), score=float(sc[qi])))

have = [r for r in rows if not np.isnan(r["dz"])]
lat = [r for r in have if r["lateral"]]
ver = [r for r in have if not r["lateral"]]
print(f"matched rotations {len(rows)}, with a nearby handle {len(have)}   "
      f"(lateral {len(lat)}, vertical {len(ver)})\n")

if not lat:
    print("no lateral hinge has a nearby predicted handle -- signal unusable")
    raise SystemExit

print("LATERAL hinges:")
print(f"{'scene':12s} {'dz/height':>10s} {'din/width':>10s} {'ratio':>7s} {'inp err':>8s}")
for r in sorted(lat, key=lambda x: -x["dz"]):
    ratio = r["dz"] / max(r["di"], 1e-6)
    print(f"{r['scene']:12s} {r['dz']:10.2f} {r['di']:10.2f} {ratio:7.2f} {r['a_inp']:8.1f}")

print("\nvertical hinges (summary):")
dzv = np.array([r["dz"] for r in ver]); div = np.array([r["di"] for r in ver])
print(f"  dz/height median {np.median(dzv):.2f}   din/width median {np.median(div):.2f}")
rv = dzv / np.maximum(div, 1e-6)
rl = np.array([r["dz"] / max(r["di"], 1e-6) for r in lat])
print(f"  ratio  lateral median {np.median(rl):.2f}   vertical median {np.median(rv):.2f}")

print("\nprecision/recall if we flip when ratio > T  (break-even precision = 50%):")
for t in (0.8, 1.0, 1.5, 2.0, 3.0):
    tp = int((rl > t).sum()); fp = int((rv > t).sum())
    prec = tp / max(tp + fp, 1)
    print(f"  T={t:4.1f}   flips {tp+fp:3d}   true {tp}/{len(lat)}   "
          f"false {fp}   precision {100*prec:5.1f}%   net {2*tp-(tp+fp):+d}")
