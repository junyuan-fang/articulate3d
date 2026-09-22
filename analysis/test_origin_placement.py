#!/usr/bin/env python
"""Redesign where the rotation origin is placed inside its own mask.

Established by gate_origin_oracle.py / diag_origin_error_axes.py:
  * the mask always contains a point within 0.25 m of the GT axis line
    (median 6 mm), so this is a placement problem, not a segmentation one;
  * the deployed "10/90 quantile endpoint, side opposite the handle" scores
    85.0%, and the best of its two candidates only 88.3% -- the rule FAMILY is
    the ceiling, not the side choice;
  * the residual error is mostly along d (the widest direction perpendicular to
    the axis), with a secondary component along e (the part's depth).

The current rule pins the origin to the mask centroid's depth, which is
mid-thickness -- but a hinge sits on a FACE, where the part meets its frame. So
widen the candidate set in both directions and measure:

    d position : quantile q or (100-q), swept
    e position : centroid, or either extreme face

Reported per variant: the deterministic score using the existing
handle-opposite side rule, and the oracle over the candidate set (the ceiling
any selector could reach). A large oracle with a small deterministic score means
the candidates are right and a selector is needed; both small means the family
is still wrong.
"""
import os
import pickle
import sys
from itertools import product

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
PKL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_leaf.pkl"
INTER = "/data/challenge/USDNet/results/inference_inter_seg_val/debug/val_preds/preds.pkl"
MASK_TH = 1e-3
ROT = 1
THRESH = 0.25
HANDLE_MARGIN = 0.12


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def dist_line(p, gorg, axis):
    a = unit(np.asarray(axis, float))
    d = np.asarray(p, float) - np.asarray(gorg, float)
    return float(np.linalg.norm(d - np.dot(d, a) * a))


def score(p, gorg, gax, pax):
    return max(dist_line(p, gorg, gax), dist_line(p, gorg, pax))


inter = pickle.load(open(INTER, "rb"))
preds = pickle.load(open(PKL, "rb"))

QS = [0.0, 2.0, 5.0, 10.0, 20.0]
ES = ["centroid", "min", "max"]
cases = []

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
        pts = pts_all[m]
        if len(pts) < 3:
            continue
        gax, gorg = gt[bid]
        u = unit(ax[qi])
        c = pts.mean(0)
        X = pts - c
        Xp = X - np.outer(X @ u, u)
        _, _, vt = np.linalg.svd(Xp, full_matrices=False)
        d = unit(vt[0] - (vt[0] @ u) * u)
        e = unit(np.cross(u, d))
        td, te = X @ d, X @ e

        near = None
        if handles is not None and len(handles):
            lo_, hi_ = pts.min(0) - HANDLE_MARGIN, pts.max(0) + HANDLE_MARGIN
            nz = handles[np.all((handles > lo_) & (handles < hi_), axis=1)]
            if len(nz):
                near = nz

        cases.append(dict(c=c, d=d, e=e, td=td, te=te, gax=gax, gorg=gorg,
                          pax=ax[qi], near=near, prev=org[qi]))

print(f"matched rotations: {len(cases)}\n")


def build(cs, q, epos):
    """Two d-candidates x one e-offset -> the candidate points for this variant."""
    lo, hi = np.percentile(cs["td"], [q, 100 - q])
    if epos == "centroid":
        eo = 0.0
    else:
        # robust quantiles, not raw min/max: the original endpoint work already
        # established that raw extremes are outlier-driven and score worse, and
        # the first version of this sweep repeated that mistake
        elo, ehi = np.percentile(cs["te"], [5, 95])
        eo = float(elo if epos == "min" else ehi)
    return [cs["c"] + t * cs["d"] + eo * cs["e"] for t in (lo, hi)]


def pick_handle(cs, cand):
    """Deployed side rule: the endpoint farther from a nearby handle."""
    if cs["near"] is None:
        k = int(np.argmin([np.linalg.norm(np.asarray(cs["prev"]) - p) for p in cand]))
        return cand[k]
    d0 = np.linalg.norm(cs["near"] - cand[0], axis=1).min()
    d1 = np.linalg.norm(cs["near"] - cand[1], axis=1).min()
    return cand[1] if d1 > d0 else cand[0]


base = np.mean([score(c["prev"], c["gorg"], c["gax"], c["pax"]) < THRESH for c in cases])
print(f"deployed (as scored today): {100*base:.1f}%\n")
print(f"{'q':>5s} {'e-pos':>9s} {'handle rule':>12s} {'oracle(2)':>10s}")
best = None
for q, epos in product(QS, ES):
    det, orc = [], []
    for cs in cases:
        cand = build(cs, q, epos)
        det.append(score(pick_handle(cs, cand), cs["gorg"], cs["gax"], cs["pax"]) < THRESH)
        orc.append(min(score(p, cs["gorg"], cs["gax"], cs["pax"]) for p in cand) < THRESH)
    row = (q, epos, 100 * np.mean(det), 100 * np.mean(orc))
    print(f"{q:5.1f} {epos:>9s} {row[2]:11.1f}% {row[3]:9.1f}%")
    if best is None or row[2] > best[2]:
        best = row

print(f"\nbest deterministic: q={best[0]} e={best[1]}  ->  {best[2]:.1f}% "
      f"(deployed {100*base:.1f}%)")

# ceiling of the whole widened family, over every (q, e-pos, side) at once
full = []
for cs in cases:
    allc = [p for q, epos in product(QS, ES) for p in build(cs, q, epos)]
    full.append(min(score(p, cs["gorg"], cs["gax"], cs["pax"]) for p in allc) < THRESH)
print(f"oracle over the ENTIRE widened family: {100*np.mean(full):.1f}%")
