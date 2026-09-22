#!/usr/bin/env python
"""Find a geometric discriminator for slide-in-plane vs slide-along-normal.

diag_inplane.py established the ceiling: choosing correctly between the panel
normal and the in-plane horizontal would give 97.9% translation-axis retention
(vs 92.6% now), and the in-plane candidate is within 0.1-0.7 deg of GT on the
cases that need it. But the model's own axis is a useless arbiter -- it sits
closer to the normal on all five, having learnt a drawers-slide-outward prior.

So dump shape features for both groups and look for a separation with a
physical story and a wide margin. A drawer front is wide and short and small; a
sliding door or window leaf is large, and usually taller than it is wide.
"""
import os
import pickle

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
RAW = "/data/challenge/USDNet/results/ev_v4init/debug/val_preds/preds.pkl"
MASK_TH = 1e-3


def ang(a, b):
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


def flatten(v):
    f = np.asarray(v, float).copy()
    f[2] = 0.0
    n = np.linalg.norm(f)
    return f / n if n > 1e-6 else None


def feats(pts):
    c = pts - pts.mean(0)
    s = c if len(c) <= 4000 else c[np.random.RandomState(0).choice(len(c), 4000, False)]
    _, sv, vt = np.linalg.svd(s, full_matrices=False)
    pl = float(sv[2] / max(sv[1], 1e-9))
    nrm = flatten(vt[2])
    if nrm is None:
        return None
    inp = np.array([-nrm[1], nrm[0], 0.0])
    w = float(np.ptp(c @ inp))          # width across the panel, horizontal
    h = float(np.ptp(pts[:, 2]))        # height
    t = float(np.ptp(c @ nrm))          # thickness through the panel
    return dict(nrm=nrm, inp=inp, pl=pl, w=w, h=h, t=t, n=len(pts),
                wh=w / max(h, 1e-6), area=w * h, zmin=float(pts[:, 2].min()),
                zmax=float(pts[:, 2].max()))


rows = []
raw = pickle.load(open(RAW, "rb"))
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
        part = pts_all[m]
        if len(part) < 10:
            continue
        f = feats(part)
        if f is None:
            continue
        f["scene"] = scene
        f["gt"] = gt[bid]
        f["mdl"] = np.asarray(raw[scene]["pred_axises"])[qi]
        f["gn"] = ang(f["gt"], f["nrm"])
        f["gi"] = ang(f["gt"], f["inp"])
        rows.append(f)

pos = [r for r in rows if r["gi"] < r["gn"]]     # slides in-plane
neg = [r for r in rows if r["gi"] >= r["gn"]]    # slides along normal
print(f"in-plane {len(pos)}   normal {len(neg)}\n")
for k in ("w", "h", "t", "wh", "area", "n", "pl", "zmin", "zmax"):
    p = np.array([r[k] for r in pos], float)
    q = np.array([r[k] for r in neg], float)
    print(f"{k:6s} in-plane med {np.median(p):8.3f} [{p.min():7.3f} {p.max():8.3f}]   "
          f"normal med {np.median(q):8.3f} [{q.min():7.3f} {q.max():8.3f}]")

print("\nin-plane cases:")
print(f"{'scene':12s} {'gt-nrm':>7s} {'gt-inp':>7s} {'w':>6s} {'h':>6s} "
      f"{'w/h':>6s} {'area':>6s} {'pl':>6s} {'n':>6s}")
for r in sorted(pos, key=lambda x: -x["gn"]):
    print(f"{r['scene']:12s} {r['gn']:7.1f} {r['gi']:7.1f} {r['w']:6.2f} "
          f"{r['h']:6.2f} {r['wh']:6.2f} {r['area']:6.2f} {r['pl']:6.3f} {r['n']:6d}")

pickle.dump(rows, open("/tmp/slider_feats.pkl", "wb"))
