#!/usr/bin/env python
"""Second sweep: the knobs left hand-picked after the quantile was optimised.

Still unswept: the canonical-snap cone (CANON_DEG=30), and the two parameters
deciding which predicted interactable counts as "the handle" for side
selection -- its score threshold (0.5) and the bbox margin around the part
(0.25 m). Side choice is now the main remaining origin lever: it is right 86.1%
of the time, so the 14% wrong-side cases are worth more than anything left in
the axis branch.

Same discipline as sweep_geo.py: report the whole curve, adopt only broad
plateaus.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "/data/challenge/USDNet")
sys.path.insert(0, "/data/challenge/analysis")

import benchmark.evaluate_semantic_instance as esi
import build_matches as bm
import postprocess2 as pp

RAW = "/data/challenge/USDNet/results/ev_v4init/debug/val_preds/preds.pkl"
PTS = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
HANDLES = "/data/challenge/USDNet/results/inference_inter_seg_val/debug/val_preds/preds.pkl"
TMP = "/tmp/_sweep2.pkl"
INTER = pickle.load(open(HANDLES, "rb"))


def handle_cols(scene, hscore):
    ip = INTER.get(scene)
    if ip is None:
        return []
    im = np.asarray(ip["pred_masks"]) > pp.MASK_THRESH
    isc = np.asarray(ip["pred_scores"]).ravel()
    return [im[:, qi].copy() for qi in np.where(isc > hscore)[0]
            if 10 <= im[:, qi].sum() < 20000]


def snap(origins, axes, classes, masks, points, hs, margin):
    origins = np.asarray(origins, dtype=np.float64).copy()
    classes = np.asarray(classes).ravel()
    masks = np.asarray(masks)
    for i in range(origins.shape[0]):
        if classes[i] != pp.ROTATION:
            continue
        sel = masks[:, i] > pp.MASK_THRESH
        if sel.sum() < 3:
            continue
        pts = points[sel]
        a = np.asarray(axes[i], dtype=np.float64)
        a = a / max(np.linalg.norm(a), 1e-9)
        c = pts.mean(axis=0)
        X = pts - c
        X = X - np.outer(X @ a, a)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        d = vt[0] - (vt[0] @ a) * a
        d = d / max(np.linalg.norm(d), 1e-9)
        t = X @ d
        lo, hi = np.percentile(t, [pp.QLO, pp.QHI])
        cand = np.stack([c + lo * d, c + hi * d])
        k = None
        if hs is not None and len(hs):
            blo, bhi = pts.min(0) - margin, pts.max(0) + margin
            near = hs[np.all((hs > blo) & (hs < bhi), axis=1)]
            if len(near):
                k = int(np.linalg.norm(near - cand[1], axis=1).min()
                        > np.linalg.norm(near - cand[0], axis=1).min())
        if k is None:
            k = int(np.argmin(np.linalg.norm(cand - origins[i], axis=1)))
        origins[i] = cand[k]
    return origins


def run(canon, hscore, margin):
    preds = pickle.load(open(RAW, "rb"))
    for scene, p in preds.items():
        pts = np.load(os.path.join(PTS, scene + ".npy"), mmap_mode="r")[:, 0:3].astype(np.float64)
        ax = pp.normalize_axes(p["pred_axises"])
        ax = pp.snap_canonical(ax, p["pred_classes"], thresh_deg=canon)
        ax, _ = pp.pca_translation_axes(ax, p["pred_classes"],
                                        np.asarray(p["pred_masks"]), pts)
        p["pred_axises"] = ax
        cols = handle_cols(scene, hscore)
        hs = np.array([pts[c].mean(0) for c in cols]) if cols else None
        p["pred_origins"] = snap(p["pred_origins"], ax, p["pred_classes"],
                                 p["pred_masks"], pts, hs, margin)
    pickle.dump(preds, open(TMP, "wb"))
    bm.PRED_PKL = TMP
    m = bm.build()[0]
    both = esi.compute_averages(esi.evaluate_matches(m, esi.match_criteria_MAO_pred_standard))["all_ap_50%"]
    ap50 = esi.compute_averages(esi.evaluate_matches(m, esi.match_criteria_M_pred))["all_ap_50%"]
    del preds
    return ap50, both


if __name__ == "__main__":
    print("current = (canon 30, hscore 0.5, margin 0.25) -> 0.4359\n")
    best = (None, 0.0)
    grid = [(c, 0.5, 0.25) for c in (15.0, 20.0, 25.0, 35.0, 40.0, 45.0)]
    grid += [(30.0, h, 0.25) for h in (0.3, 0.4, 0.6, 0.7)]
    grid += [(30.0, 0.5, m) for m in (0.10, 0.15, 0.35, 0.50)]
    for canon, hscore, margin in grid:
        ap50, both = run(canon, hscore, margin)
        star = ""
        if both > best[1]:
            best = ((canon, hscore, margin), both)
            star = "  *"
        print(f"  canon={canon:5.1f} hscore={hscore:.1f} margin={margin:.2f}  "
              f"AP50={ap50:.4f}  BOTH={both:.4f}{star}", flush=True)
    print("\nbest:", best)
