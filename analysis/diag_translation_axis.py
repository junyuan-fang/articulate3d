#!/usr/bin/env python
"""Where exactly do the remaining translation axes fail?

Translation axis retention (91.6% on val) is the only factor where we trail the
leader; rotation axis is already at its ceiling (all 12 of its failures are
>20 deg off -- the lateral hinges the probes proved unfixable). So this is the
last movable multiplier in both = AP50 x r_axis x r_origin.

For each matched translation we report the deployed error alongside the two
estimators that produced it, plus the part's planarity -- the PCA-normal rule
assumes a flat front panel, and that assumption is the obvious suspect.
"""
import os
import pickle

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
RAW = "/data/challenge/USDNet/results/ev_v4init/debug/val_preds/preds.pkl"
FINAL = "/tmp/final_val.pkl"
MASK_TH = 1e-3


def ang(a, b):
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


def flat_pca(pts):
    c = pts - pts.mean(0)
    if len(c) > 4000:
        c = c[np.random.RandomState(0).choice(len(c), 4000, replace=False)]
    _, sv, vt = np.linalg.svd(c, full_matrices=False)
    return vt, sv


raw = pickle.load(open(RAW, "rb"))
fin = pickle.load(open(FINAL, "rb"))
rows = []
for scene in sorted(raw):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    pts_all = np.asarray(a[:, 0:3])
    inst = np.asarray(a[:, 10]).astype(int)
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            g = h[k]
            if int(np.asarray(g["sem_id"])) == 2:
                gt[int(k)] = np.asarray(g["axis"], float)
    if not gt:
        continue
    gmasks = {i: inst == i for i in gt}
    pm = np.asarray(raw[scene]["pred_masks"]) > MASK_TH
    sc = np.asarray(raw[scene]["pred_scores"]).ravel()
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
        part = pts_all[m]
        if len(part) < 10:
            continue
        vt, sv = flat_pca(part)
        planarity = float(sv[2] / max(sv[1], 1e-9))
        model_ax = np.asarray(raw[scene]["pred_axises"])[qi]
        final_ax = np.asarray(fin[scene]["pred_axises"])[qi]
        geo = None
        for v in (vt[2], vt[1]):
            f = v.copy()
            f[2] = 0.0
            n = np.linalg.norm(f)
            if n > 0.3:
                geo = f / n
                break
        rows.append((scene, ang(final_ax, gax), ang(model_ax, gax),
                     ang(geo, gax) if geo is not None else np.nan,
                     planarity, int(m.sum())))

R = rows
fails = [r for r in R if r[1] >= 15]
print(f"matched translations {len(R)}   axis failures {len(fails)}   "
      f"pass {100*(1-len(fails)/len(R)):.1f}%\n")
print(f"{'scene':12s} {'final':>7s} {'model':>7s} {'pca':>7s} {'planarity':>10s} {'pts':>7s}")
for r in sorted(fails, key=lambda x: x[1]):
    print(f"{r[0]:12s} {r[1]:7.1f} {r[2]:7.1f} {r[3]:7.1f} {r[4]:10.3f} {r[5]:7d}")

ok = [r for r in R if r[1] < 15]
print(f"\nplanarity  failures median {np.median([r[4] for r in fails]):.3f}   "
      f"passes median {np.median([r[4] for r in ok]):.3f}")
print(f"pts        failures median {np.median([r[5] for r in fails]):.0f}   "
      f"passes median {np.median([r[5] for r in ok]):.0f}")
need = int(np.floor(len(R) * 0.05))
print(f"\nfor 95% retention: at most {need} failures allowed "
      f"(currently {len(fails)}) -> must fix {max(0, len(fails)-need)}")
