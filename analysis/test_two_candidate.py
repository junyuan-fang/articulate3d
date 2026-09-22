#!/usr/bin/env python
"""Translation axis: two physical candidates, arbitrated by the model.

Diagnosis (analysis/diag_translation_axis.py) shows the current rule fails in
two ways, both from the same wrong assumption -- that a sliding part always
moves along its panel normal:

  * five failures sit at ~89.5 deg from the PCA normal on very planar parts
    (planarity 0.03-0.04). Off by exactly a right angle means the GT direction
    lies IN the panel plane: these are sliding doors/windows, which travel
    parallel to their own face, not drawers travelling through it.
  * one failure (88627b561e) has a PCA normal 0.2 deg from GT, rejected because
    it disagreed with the model by more than the 25 deg gate. Using the model as
    a veto throws away a near-perfect estimate.

Proposal: for planar parts propose BOTH candidates (normal, in-plane
horizontal) and let the model's own axis pick the nearer one -- arbitration
instead of veto. Non-planar parts keep the model axis, since PCA is meaningless
there (the two 15-17 deg failures have planarity ~0.5).
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


def analyse(pts):
    c = pts - pts.mean(0)
    if len(c) > 4000:
        c = c[np.random.RandomState(0).choice(len(c), 4000, replace=False)]
    _, sv, vt = np.linalg.svd(c, full_matrices=False)
    planarity = float(sv[2] / max(sv[1], 1e-9))
    normal = flatten(vt[2])
    inplane = np.array([-normal[1], normal[0], 0.0]) if normal is not None else None
    return normal, inplane, planarity


def collect():
    raw = pickle.load(open(RAW, "rb"))
    out = []
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
            mdl = np.asarray(raw[scene]["pred_axises"])[qi]
            n_, ip_, pl = analyse(part)
            out.append((gt[bid], mdl, n_, ip_, pl))
    return out


if __name__ == "__main__":
    rows = collect()
    print(f"matched translations: {len(rows)}\n")

    def rate(errs):
        e = np.array(errs)
        return f"{np.mean(e < 15)*100:5.1f}%  median {np.median(e):5.2f}"

    model = [ang(m, g) for g, m, _, _, _ in rows]
    print(f"model only            {rate(model)}")

    cur = []
    for g, m, n_, ip_, pl in rows:
        a = n_ if (n_ is not None and ang(m, n_) < 25) else m
        cur.append(ang(a, g))
    print(f"current (gate 25)     {rate(cur)}")

    for th in (0.06, 0.10, 0.15, 0.20, 0.30):
        new = []
        for g, m, n_, ip_, pl in rows:
            mh = flatten(m)
            if mh is None:
                mh = m / max(np.linalg.norm(m), 1e-9)
            if n_ is not None and pl < th:
                a = min([n_, ip_], key=lambda c: ang(mh, c))
            elif n_ is not None and ang(m, n_) < 25:
                a = n_
            else:
                a = m
            new.append(ang(a, g))
        print(f"two-candidate pl<{th:.2f}  {rate(new)}")
