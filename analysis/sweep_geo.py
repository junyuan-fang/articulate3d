#!/usr/bin/env python
"""Sweep the three geometric-stack hyperparameters against val.

Re-diagnosis of the current pipeline (analysis/rediagnose_final.py) shows the
残 loss is dominated by rotation origins (23 origin-only failures vs 5
axis-only), with four of them at 0.306-0.340 m against a 0.25 m threshold and
two translation axes at 15.2/16.8 deg against 15 deg. All three knobs below
were set by coarse hand-picking and never swept.

`outward` is the new one: a door's predicted mask tends to stop short of the
hinge edge (the frame occludes the seam), so pushing the chosen endpoint a few
centimetres further out along the same direction should remove a systematic
bias -- exactly the shape of those 0.30-0.34 m near-misses.

Reports val AP50_axis_origin for each setting so sensitivity (not just the
argmax) is visible; a knob whose neighbourhood is flat is safe to adopt, a
spike is val overfitting.
"""
import itertools
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
TMP = "/tmp/_sweep.pkl"


def load_handles():
    inter = pickle.load(open(HANDLES, "rb"))
    out = {}
    for scene, ip in inter.items():
        im = np.asarray(ip["pred_masks"]) > pp.MASK_THRESH
        isc = np.asarray(ip["pred_scores"]).ravel()
        cols = [im[:, qi].copy() for qi in np.where(isc > 0.5)[0]
                if 10 <= im[:, qi].sum() < 20000]
        out[scene] = cols
    return out


def snap_endpoints(origins, axes, classes, masks, points, handles,
                   qlo, qhi, outward):
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
        lo, hi = np.percentile(t, [qlo, qhi])
        cand = np.stack([c + (lo - outward) * d, c + (hi + outward) * d])
        k = None
        hs = handles.get("_cur")
        if hs is not None and len(hs):
            blo, bhi = pts.min(0) - 0.25, pts.max(0) + 0.25
            near = hs[np.all((hs > blo) & (hs < bhi), axis=1)]
            if len(near):
                k = int(np.linalg.norm(near - cand[1], axis=1).min()
                        > np.linalg.norm(near - cand[0], axis=1).min())
        if k is None:
            k = int(np.argmin(np.linalg.norm(cand - origins[i], axis=1)))
        origins[i] = cand[k]
    return origins


def build(qlo, qhi, outward, gate, hcols):
    preds = pickle.load(open(RAW, "rb"))
    for scene, p in preds.items():
        pts = np.load(os.path.join(PTS, scene + ".npy"), mmap_mode="r")[:, 0:3].astype(np.float64)
        ax = pp.normalize_axes(p["pred_axises"])
        ax = pp.snap_canonical(ax, p["pred_classes"])
        if gate > 0:
            ax, _ = pp.pca_translation_axes(ax, p["pred_classes"],
                                            np.asarray(p["pred_masks"]), pts, agree_deg=gate)
        p["pred_axises"] = ax
        cols = hcols.get(scene, [])
        hs = np.array([pts[c].mean(0) for c in cols]) if cols else None
        p["pred_origins"] = snap_endpoints(p["pred_origins"], ax, p["pred_classes"],
                                           p["pred_masks"], pts, {"_cur": hs},
                                           qlo, qhi, outward)
    pickle.dump(preds, open(TMP, "wb"))
    bm.PRED_PKL = TMP
    matches, _, _ = bm.build()
    both = esi.compute_averages(esi.evaluate_matches(
        matches, esi.match_criteria_MAO_pred_standard))["all_ap_50%"]
    ap50 = esi.compute_averages(esi.evaluate_matches(
        matches, esi.match_criteria_M_pred))["all_ap_50%"]
    del preds
    return ap50, both


if __name__ == "__main__":
    hcols = load_handles()
    print("baseline stack = (5, 95, outward 0.00, gate 25) -> 0.4306\n")
    grid = [(q, 100 - q, 0.0, 25.0) for q in (10, 12, 15, 20, 25, 30)]
    grid += [(15, 85, 0.0, g) for g in (20.0, 30.0)]
    best = (None, 0.0)
    for (qlo, qhi, outward, gate) in grid:
        ap50, both = build(qlo, qhi, outward, gate, hcols)
        star = ""
        if both > best[1]:
            best = ((qlo, qhi, outward, gate), both)
            star = "  *"
        print(f"  q=({qlo:2d},{qhi:2d}) outward={outward:.2f} gate={gate:5.1f}  "
              f"AP50={ap50:.4f}  BOTH={both:.4f}{star}", flush=True)
    print("\nbest:", best)
