#!/usr/bin/env python
"""Can a learned selector beat the hand-written one at choosing the hinge side?

Everything left in origin is selection: the candidate set contains a passing
point (2-endpoint oracle 88.3%, widened family 95.6%, mask ceiling 100%) but no
deterministic rule we have tried exceeds 85.6%, and three different selectors --
handle-opposite, nearest-to-model-origin, and their combination -- all score
exactly 85.0%.

Before spending an inference run over the 195 train scenes, check on validation
whether ANY learnable signal exists, using leave-one-scene-out CV so the number
is honest, plus a label-permutation null so we can tell skill from chance. The
selector's job is a balanced binary choice, which is a far better-posed problem
than the 3.4%-prevalence lateral-hinge detection that defeated eleven detectors.

Features are deliberately geometric and exclude the model's own predicted
origin: that head was trained on the train split, so a feature built from it
would be scored optimistically there and mislead the weighting.
"""
import os
import pickle
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
PKL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_leaf.pkl"
INTER = "/data/challenge/USDNet/results/inference_inter_seg_val/debug/val_preds/preds.pkl"
MASK_TH = 1e-3
ROT = 1
THRESH = 0.25
QLO, QHI = 10.0, 90.0
HANDLE_MARGIN = 0.12


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def dist_line(p, gorg, axis):
    a = unit(np.asarray(axis, float))
    d = np.asarray(p, float) - np.asarray(gorg, float)
    return float(np.linalg.norm(d - np.dot(d, a) * a))


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
                gt[int(k)] = (np.asarray(g["axis"], float),
                              np.asarray(g["origin"], float))
    if not gt:
        continue
    tree = cKDTree(pts_all)

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
        t = X @ d
        lo, hi = np.percentile(t, [QLO, QHI])
        cand = [c + lo * d, c + hi * d]
        width = float(np.ptp(t))
        errs = [max(dist_line(p, gorg, gax), dist_line(p, gorg, ax[qi])) for p in cand]

        feats = []
        for k, p in enumerate(cand):
            # scene geometry around the candidate, excluding the part itself:
            # a hinge edge abuts the static frame, a free edge does not
            near_idx = tree.query_ball_point(p, 0.10)
            near_all = len(near_idx)
            part_set = set(np.flatnonzero(m).tolist())
            near_other = sum(1 for i in near_idx if i not in part_set)
            hd = np.nan
            if handles is not None and len(handles):
                lo_, hi_ = pts.min(0) - HANDLE_MARGIN, pts.max(0) + HANDLE_MARGIN
                nz = handles[np.all((handles > lo_) & (handles < hi_), axis=1)]
                if len(nz):
                    hd = float(np.linalg.norm(nz - p, axis=1).min())
            feats.append(dict(
                other_density=near_other / max(near_all, 1),
                other_count=np.log1p(near_other),
                handle_dist=hd if hd == hd else 2 * width,
                z=float(p[2]),
                dist_to_wallish=float(np.mean(np.abs((pts_all[near_idx] - p) @ d)))
                if near_all else 0.0,
            ))
        rows.append(dict(scene=scene, feats=feats, errs=errs, width=width))

print(f"matched rotations with 2 candidates: {len(rows)}")
solvable = [r for r in rows if min(r["errs"]) < THRESH]
print(f"of which at least one candidate passes: {len(solvable)} "
      f"({100*len(solvable)/len(rows):.1f}%  <- the 2-candidate oracle)\n")

FEATS = ["other_density", "other_count", "handle_dist", "z", "dist_to_wallish"]


def vec(r):
    """Difference of features between the two candidates -> binary problem."""
    return np.array([r["feats"][1][f] - r["feats"][0][f] for f in FEATS])


X = np.array([vec(r) for r in rows])
y = np.array([int(np.argmin(r["errs"])) for r in rows])
scenes = np.array([r["scene"] for r in rows])
passes = np.array([[e < THRESH for e in r["errs"]] for r in rows])

mu, sd = X.mean(0), X.std(0) + 1e-9
Xn = (X - mu) / sd

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    print("sklearn unavailable"); raise SystemExit


def loso(y_):
    pred = np.zeros(len(y_), int)
    for s in np.unique(scenes):
        te = scenes == s
        tr = ~te
        if len(np.unique(y_[tr])) < 2:
            pred[te] = 0
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xn[tr], y_[tr])
        pred[te] = clf.predict(Xn[te])
    return pred


pred = loso(y)
sel_pass = passes[np.arange(len(rows)), pred]
print(f"deployed handle rule                     : 85.0%")
print(f"learned selector, leave-one-scene-out CV : {100*sel_pass.mean():.1f}%")
print(f"2-candidate oracle                       : {100*np.mean(passes.any(1)):.1f}%")

rng = np.random.RandomState(0)
null = []
for _ in range(20):
    yp = rng.permutation(y)
    p = loso(yp)
    null.append(passes[np.arange(len(rows)), p].mean())
null = np.array(null)
print(f"\npermutation null (20 shuffles): {100*null.mean():.1f}% +- {100*null.std():.1f}")
print(f"selector is {(sel_pass.mean()-null.mean())/max(null.std(),1e-9):.1f} sigma above chance")
