#!/usr/bin/env python
"""If handle prediction were perfect, how much would the origin improve?

This is the ceiling of the one remaining training idea: the handle is the only
cue with real signal for the hinge side (AUC 0.820,
inventory_hinge_signals.py), and the ground truth carries an `inter_mask` per
articulation instance -- the handle-to-part PAIRING -- which we have never used
for anything. Option (2) is to train the articulation heads with that pairing as
auxiliary supervision, in the hope the features grow a notion of where the hinge
edge is.

Worth doing only if perfect handles would actually buy something. So substitute
the GT `inter_mask` centroid for the companion model's predicted handle, keep
everything else identical, and measure:

    deployed (predicted handles)      what we score now
    GT handles                        ceiling of any handle improvement
    2-candidate oracle                ceiling of the endpoint family itself

If GT handles land near the deployed number, better handles are worthless and
option (2) is dead. If they land near the oracle, handle quality is the binding
constraint and the auxiliary task is aimed correctly.
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
res = {"deployed (predicted handles)": [], "GT handle of THIS part": [],
       "GT handles, any part": [], "2-candidate oracle": []}
n_gt_handle = n_total = 0

for scene in sorted(preds):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    pts_all = np.asarray(a[:, 0:3])
    inst = np.asarray(a[:, 10]).astype(int)
    gt, gt_handle = {}, {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            g = h[k]
            if int(np.asarray(g["sem_id"])) != ROT:
                continue
            gt[int(k)] = (np.asarray(g["axis"], float), np.asarray(g["origin"], float))
            # inter_mask is indexed over THIS PART's own points, not the scene:
            # its length equals the part's point count. Indexing the scene with
            # it yields unrelated points -- which is exactly the bug that made
            # ground-truth handles look worse than predicted ones.
            if "inter_mask" in g:
                im = np.asarray(g["inter_mask"]).ravel().astype(bool)
                part_idx = np.flatnonzero(inst == int(k))
                if len(im) == len(part_idx) and im.any():
                    gt_handle[int(k)] = pts_all[part_idx[im]].mean(0)
    if not gt:
        continue

    pred_h = []
    if scene in inter:
        im = np.asarray(inter[scene]["pred_masks"]) > MASK_TH
        isc = np.asarray(inter[scene]["pred_scores"]).ravel()
        for hi in np.where(isc > 0.5)[0]:
            sel = im[:, hi]
            if 10 <= sel.sum() < 20000:
                pred_h.append(pts_all[sel].mean(0))
    pred_h = np.array(pred_h) if pred_h else None
    all_gt_h = np.array(list(gt_handle.values())) if gt_handle else None

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
        t = X @ d
        lo, hi = np.percentile(t, [QLO, QHI])
        cand = [c + lo * d, c + hi * d]
        errs = [max(dist_line(p, gorg, gax), dist_line(p, gorg, ax[qi])) for p in cand]

        def choose(hs, restrict_bbox=True):
            if hs is None or not len(hs):
                k = int(np.argmin([np.linalg.norm(np.asarray(org[qi]) - p) for p in cand]))
                return cand[k]
            hs = np.atleast_2d(hs)
            if restrict_bbox:
                blo, bhi = pts.min(0) - HANDLE_MARGIN, pts.max(0) + HANDLE_MARGIN
                nz = hs[np.all((hs > blo) & (hs < bhi), axis=1)]
                if not len(nz):
                    k = int(np.argmin([np.linalg.norm(np.asarray(org[qi]) - p) for p in cand]))
                    return cand[k]
                hs = nz
            d0 = np.linalg.norm(hs - cand[0], axis=1).min()
            d1 = np.linalg.norm(hs - cand[1], axis=1).min()
            return cand[1] if d1 > d0 else cand[0]

        n_total += 1
        own = gt_handle.get(bid)
        if own is not None:
            n_gt_handle += 1

        res["deployed (predicted handles)"].append(
            max(dist_line(choose(pred_h), gorg, gax), dist_line(choose(pred_h), gorg, ax[qi])) < THRESH)
        p_own = choose(own[None, :], restrict_bbox=False) if own is not None else choose(pred_h)
        res["GT handle of THIS part"].append(
            max(dist_line(p_own, gorg, gax), dist_line(p_own, gorg, ax[qi])) < THRESH)
        p_any = choose(all_gt_h)
        res["GT handles, any part"].append(
            max(dist_line(p_any, gorg, gax), dist_line(p_any, gorg, ax[qi])) < THRESH)
        res["2-candidate oracle"].append(min(errs) < THRESH)

print(f"matched rotations: {n_total}   with a GT handle of their own: {n_gt_handle} "
      f"({100*n_gt_handle/max(n_total,1):.0f}%)\n")
for k, v in res.items():
    print(f"  {k:30s} {100*np.mean(v):5.1f}%")
