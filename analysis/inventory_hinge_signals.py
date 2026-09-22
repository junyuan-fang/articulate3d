#!/usr/bin/env python
"""Which raw signal, if any, tells the hinge edge from the free edge?

Reframing: origin and axis are one problem. A door's hinge edge determines BOTH
-- the rotation axis runs along it, and the origin is any point on it. We have
been solving them separately (axis from canonical/anchor priors, origin from
endpoints plus the handle) and both are now stuck on selection, not on
representation.

Two inference-time signals have never been used at all: RGB (columns 3-5) and
the surface normals (columns 6-8, used only for superpoints). The companion
interactable model is also under-used -- we take mask centroids and throw the
geometry away.

Physically, a hinge edge differs from a free edge in ways those signals should
see:

  contact   the hinge edge touches the static frame; the free edge opens onto a
            gap, so scene points cease just beyond it
  crease    at the hinge the door meets the frame at an angle -- a normal
            discontinuity the free edge lacks
  colour    hinges are hardware, usually metal, against a painted panel
  handle    the known baseline: the handle sits at the far edge

This measures each cue's own AUC for "is this the hinge edge", before any
combination, with a permutation null. Everything is computed along the whole
EDGE LINE rather than in a ball around one endpoint -- measuring at a single
point is what made the previous attempt uninformative.
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
NSAMP = 24
RAD = 0.06


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def dist_line(p, gorg, axis):
    a = unit(np.asarray(axis, float))
    d = np.asarray(p, float) - np.asarray(gorg, float)
    return float(np.linalg.norm(d - np.dot(d, a) * a))


def auc(scores, labels):
    """Probability a positive outranks a negative; 0.5 is chance."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s));  ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


inter = pickle.load(open(INTER, "rb"))
preds = pickle.load(open(PKL, "rb"))
FEATS = ["contact", "gap_beyond", "crease", "colour_diff", "colour_var", "handle_dist"]
recs = []

for scene in sorted(preds):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    pts_all = np.asarray(a[:, 0:3])
    rgb_all = np.asarray(a[:, 3:6])
    nrm_all = np.asarray(a[:, 6:9])
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
        idx_part = np.flatnonzero(m)
        pts = pts_all[idx_part]
        if len(pts) < 50:
            continue
        part_set = set(idx_part.tolist())
        gax, gorg = gt[bid]

        u = unit(ax[qi])
        c = pts.mean(0)
        X = pts - c
        Xp = X - np.outer(X @ u, u)
        _, _, vt = np.linalg.svd(Xp, full_matrices=False)
        d = unit(vt[0] - (vt[0] @ u) * u)
        t = X @ d
        lo, hi = np.percentile(t, [QLO, QHI])
        along = X @ u
        s_lo, s_hi = np.percentile(along, [5, 95])
        part_normal = unit(nrm_all[idx_part].mean(0))
        part_rgb = rgb_all[idx_part].mean(0)

        cands, feats = [], []
        for tk, sgn in ((lo, -1.0), (hi, +1.0)):
            base = c + tk * d
            cands.append(base)
            samples = [base + s * u for s in np.linspace(s_lo, s_hi, NSAMP)]
            contact, beyond, crease, cdiff, cvar = [], [], [], [], []
            for p in samples:
                nb = tree.query_ball_point(p, RAD)
                other = [i for i in nb if i not in part_set]
                contact.append(len(other))
                # emptiness just past the edge: a free edge opens onto a gap
                nb2 = tree.query_ball_point(p + sgn * 0.05 * d, RAD)
                beyond.append(len([i for i in nb2 if i not in part_set]))
                if other:
                    on = nrm_all[other]
                    on = on / np.maximum(np.linalg.norm(on, axis=1, keepdims=True), 1e-9)
                    crease.append(float(np.mean(1.0 - np.abs(on @ part_normal))))
                    cdiff.append(float(np.linalg.norm(rgb_all[other].mean(0) - part_rgb)))
                mine = [i for i in nb if i in part_set]
                if len(mine) > 3:
                    cvar.append(float(rgb_all[mine].std(0).mean()))
            hd = np.nan
            if handles is not None and len(handles):
                hd = float(np.linalg.norm(handles - base, axis=1).min())
            feats.append(dict(
                contact=float(np.mean(contact)),
                gap_beyond=-float(np.mean(beyond)),   # more emptiness -> free edge
                crease=float(np.mean(crease)) if crease else 0.0,
                colour_diff=float(np.mean(cdiff)) if cdiff else 0.0,
                colour_var=float(np.mean(cvar)) if cvar else 0.0,
                handle_dist=hd if hd == hd else 0.0,
            ))
        errs = [max(dist_line(p, gorg, gax), dist_line(p, gorg, ax[qi])) for p in cands]
        if min(errs) >= THRESH:
            continue                      # neither candidate is the hinge; skip
        recs.append(dict(scene=scene, feats=feats, hinge=int(np.argmin(errs))))

print(f"instances where one candidate IS the hinge edge: {len(recs)}\n")
print(f"{'signal':>14s} {'AUC':>7s}   {'reading':<44s}")
rng = np.random.RandomState(0)
for f in FEATS:
    s, y = [], []
    for r in recs:
        s += [r["feats"][0][f], r["feats"][1][f]]
        y += [r["hinge"] == 0, r["hinge"] == 1]
    a_ = auc(s, y)
    null = np.std([auc(s, rng.permutation(y)) for _ in range(50)])
    sig = abs(a_ - 0.5) / max(null, 1e-9)
    note = ("informative" if sig > 3 else "chance" if sig < 2 else "weak")
    print(f"{f:>14s} {a_:7.3f}   {note} ({sig:.1f} sigma from 0.5)")

print("\nPer-instance accuracy if we pick the candidate maximising each signal:")
for f in FEATS:
    hit = [int(np.argmax([r["feats"][0][f], r["feats"][1][f]]) == r["hinge"]) for r in recs]
    print(f"  {f:>14s} {100*np.mean(hit):5.1f}%")
print(f"  {'handle (min)':>14s} "
      f"{100*np.mean([int(np.argmin([r['feats'][0]['handle_dist'], r['feats'][1]['handle_dist']]) != r['hinge']) for r in recs]):5.1f}%"
      "   <- the deployed rule: hinge is FAR from the handle")
