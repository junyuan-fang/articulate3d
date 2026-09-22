#!/usr/bin/env python
"""Do the frozen backbone features know where the hinge is?

This is the test behind "can we train a new head instead of retraining
segmentation". Hand-crafted cues are exhausted -- inventory_hinge_signals.py
found RGB, normals, crease and contact all at chance, with only the handle
carrying signal. But that says nothing about what the NETWORK's learned
representation encodes.

An earlier probe concluded the frozen features carry no articulation signal.
That conclusion is not safe: it was run on lateral-hinge classification with
SEVEN positives, where an AUC has almost no statistical power. The origin
problem is far better posed, so re-ask the question properly.

The features are per-query -- a 256-d vector of [query embedding, mask-pooled
point features], the exact input to _axis_bin_residual. Being per-query they are
symmetric in the two candidate edges, and the sign of d comes from an SVD and is
arbitrary, so "which side" is not even well defined against them. Two
formulations that are:

  A. classification, handle-relative: does the deployed "hinge opposite the
     handle" rule pick WRONG here? A working probe would let us flip exactly
     those cases. Ceiling +3.3 points (85.0% -> 88.3%).

  B. regression, canonical: where along the width does the true hinge sit,
     expressed as a fraction of the part's own extent and signed so that 0 is
     the handle end and 1 the far end? This is sign-free and asks the stronger
     question -- not "which of two" but "where". Its ceiling is the widened
     family's 95.6%.

Both under leave-one-scene-out CV against a permutation null.
"""
import os
import pickle
import sys

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
PKL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_leaf.pkl"
INTER = "/data/challenge/USDNet/results/inference_inter_seg_val/debug/val_preds/preds.pkl"
FEATS = "/data/challenge/analysis/probe_feats_matched.pkl"
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


_raw = pickle.load(open(FEATS, "rb"))
# The dump's scene order is not sorted(), but its per-scene counts match the
# matching below exactly, so group by scene and consume in order within each.
feat_by_scene = {}
for _s, _v, _l in _raw:
    feat_by_scene.setdefault(_s, []).append(np.asarray(_v, float))
inter = pickle.load(open(INTER, "rb"))
preds = pickle.load(open(PKL, "rb"))

recs = []
for scene in sorted(preds):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    pts_all = np.asarray(a[:, 0:3])
    inst = np.asarray(a[:, 10]).astype(int)
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for kk in h.keys():
            g = h[kk]
            if int(np.asarray(g["sem_id"])) == ROT:
                gt[int(kk)] = (np.asarray(g["axis"], float),
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
    k = 0
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
        bucket = feat_by_scene.get(scene)
        assert bucket, f"no probe features for {scene}"
        fvec = bucket[k]
        k += 1
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

        # canonicalise d so it points AWAY from the handle; without this the
        # sign is an SVD artefact and no label defined against it can be learnt
        hd = None
        if handles is not None and len(handles):
            lo_, hi_ = pts.min(0) - HANDLE_MARGIN, pts.max(0) + HANDLE_MARGIN
            nz = handles[np.all((handles > lo_) & (handles < hi_), axis=1)]
            if len(nz):
                hd = nz[np.argmin(np.linalg.norm(nz - c, axis=1))]
        if hd is None:
            continue                       # no handle: the frame is undefined
        if np.dot(hd - c, d) > 0:
            d, t, lo, hi = -d, -t, -hi, -lo

        cand = [c + lo * d, c + hi * d]    # cand[0] = handle end, cand[1] = far end
        errs = [max(dist_line(p, gorg, gax), dist_line(p, gorg, ax[qi])) for p in cand]
        if min(errs) >= THRESH:
            continue

        # where along the width the hinge really is, 0 = handle end, 1 = far end
        span = max(hi - lo, 1e-6)
        proj = (X @ d)
        near = np.argmin([dist_line(p, gorg, gax) for p in pts[::max(1, len(pts)//400)]])
        t_true = float(proj[::max(1, len(pts)//400)][near])
        recs.append(dict(scene=scene, f=np.asarray(fvec, float),
                         flip=int(np.argmin(errs) == 0),      # 1 = handle rule wrong
                         pos=float(np.clip((t_true - lo) / span, -0.5, 1.5))))

print(f"instances usable (matched, has handle, solvable): {len(recs)}")
y_flip = np.array([r["flip"] for r in recs])
print(f"handle rule wrong on: {int(y_flip.sum())}/{len(recs)} "
      f"({100*y_flip.mean():.1f}%  <- what a probe would have to find)\n")

X = np.stack([r["f"] for r in recs])
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
scenes = np.array([r["scene"] for r in recs])
pos = np.array([r["pos"] for r in recs])

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score


def loso_clf(y, C=0.05):
    p = np.zeros(len(y))
    for s in np.unique(scenes):
        te = scenes == s
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        p[te] = LogisticRegression(max_iter=3000, C=C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def loso_reg(y, alpha=100.0):
    p = np.zeros(len(y))
    for s in np.unique(scenes):
        te = scenes == s
        tr = ~te
        p[te] = Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[te])
    return p


rng = np.random.RandomState(0)
p = loso_clf(y_flip)
a = roc_auc_score(y_flip, p)
null = [roc_auc_score(y_flip, loso_clf(rng.permutation(y_flip))) for _ in range(15)]
print(f"A. 'handle rule is wrong here'  LOSO AUC {a:.3f}   "
      f"null {np.mean(null):.3f}+-{np.std(null):.3f}   "
      f"{abs(a-np.mean(null))/max(np.std(null),1e-9):.1f} sigma")

pr = loso_reg(pos)
ss_res = float(((pos - pr) ** 2).sum())
ss_tot = float(((pos - pos.mean()) ** 2).sum())
r2 = 1 - ss_res / max(ss_tot, 1e-9)
nulls = []
for _ in range(15):
    yp = rng.permutation(pos)
    q = loso_reg(yp)
    nulls.append(1 - float(((yp - q) ** 2).sum()) / max(float(((yp - yp.mean()) ** 2).sum()), 1e-9))
print(f"B. 'where along the width'      LOSO R2  {r2:.3f}   "
      f"null {np.mean(nulls):.3f}+-{np.std(nulls):.3f}   "
      f"{abs(r2-np.mean(nulls))/max(np.std(nulls),1e-9):.1f} sigma")
print(f"   (true positions: median {np.median(pos):.2f}, "
      f"share beyond the two endpoints {100*np.mean((pos<0)|(pos>1)):.0f}%)")
