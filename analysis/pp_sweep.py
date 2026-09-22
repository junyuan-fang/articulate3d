#!/usr/bin/env python
"""Sweep training-free origin/axis post-processing variants and score each one
with the official val criteria, loading the 5 GB preds.pkl only once.

Motivation: postprocess2.py's `endpoint` step (snap origin to the 2/98 pct
endpoints of the mask along its *first* PCA direction) actually destroys the
origin score end to end (AP50_origin 0.4495 -> 0.2619). The oracle in
docs/05-improvement.md conditioned on the GT axis, and for a door hinged about a
vertical axis the longest extent is vertical while the hinge sits on a *lateral*
edge -- i.e. an endpoint along the direction perpendicular to the axis. This
script tests that reading.

Usage: python pp_sweep.py [preds.pkl]
"""
import sys
import pickle
from pathlib import Path

import numpy as np

import build_matches as bm
import benchmark.evaluate_semantic_instance as esi

POINTS_DIR = Path("/data/challenge/USDNet/data/processed/"
                  "articulate3d_challenge_mov/validation")
Z = np.array([0.0, 0.0, 1.0])
ROTATION, TRANSLATION = 1, 2
MASK_THRESH = 0.5


def normalize(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


def snap_canonical(axes, classes, thresh_deg=30.0):
    axes = normalize(np.asarray(axes, np.float64)).copy()
    classes = np.asarray(classes).ravel()
    cos_z = np.abs(axes @ Z)
    lim = np.cos(np.deg2rad(thresh_deg))
    axes[(classes == ROTATION) & (cos_z >= lim)] = Z
    tr = (classes == TRANSLATION) & (cos_z <= np.cos(np.deg2rad(90 - thresh_deg)))
    if tr.any():
        flat = axes[tr].copy()
        flat[:, 2] = 0.0
        nrm = np.linalg.norm(flat, axis=1, keepdims=True).ravel()
        keep = nrm > 1e-6
        idx = np.where(tr)[0][keep]
        axes[idx] = flat[keep] / nrm[keep][:, None]
    return axes


def _endpoints(pts, direction, qlo, qhi):
    c = pts.mean(0)
    t = (pts - c) @ direction
    lo, hi = np.percentile(t, [qlo, qhi])
    return np.stack([c + lo * direction, c + hi * direction])


def snap_origin(origins, axes, classes, masks, points, mode, qlo=2.0, qhi=98.0):
    """mode: 'pca'  -> endpoints along the 1st PCA direction of the mask
             'perp' -> endpoints along the max-extent direction PERPENDICULAR
                       to the predicted axis (the hinge side of a door)
             'near' -> nearest point of the mask to the current origin
    """
    origins = np.asarray(origins, np.float64).copy()
    classes = np.asarray(classes).ravel()
    masks = np.asarray(masks)
    moved = 0
    for i in range(origins.shape[0]):
        if classes[i] != ROTATION:
            continue
        sel = masks[:, i] > MASK_THRESH
        if sel.sum() < 3:
            continue
        pts = points[sel]
        if mode == "near":
            origins[i] = pts[np.argmin(np.linalg.norm(pts - origins[i], axis=1))]
            moved += 1
            continue
        c = pts.mean(0)
        X = pts - c
        if mode == "perp":
            a = normalize(np.asarray(axes[i], np.float64))
            X = X - np.outer(X @ a, a)          # project onto plane perp to axis
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        d = vt[0]
        if mode == "perp":
            a = normalize(np.asarray(axes[i], np.float64))
            d = normalize(d - (d @ a) * a)
        cand = _endpoints(pts, d, qlo, qhi)
        origins[i] = cand[int(np.argmin(np.linalg.norm(cand - origins[i], axis=1)))]
        moved += 1
    return origins, moved


def score(preds, tag):
    bm._PREBUILT = preds
    matches = build_matches_from(preds)
    row = [tag]
    for crit in (esi.match_criteria_M_pred, esi.match_criteria_MA_pred,
                 esi.match_criteria_MO_pred, esi.match_criteria_MAO_pred_standard):
        ap = esi.compute_averages(esi.evaluate_matches(matches, crit))
        row.append(ap["all_ap_50%"])
    print(f"{row[0]:38s}  AP50 {row[1]:.4f}  axis {row[2]:.4f}  "
          f"origin {row[3]:.4f}  BOTH {row[4]:.4f}", flush=True)
    return row


def build_matches_from(preds):
    import os
    bm.set_articulate3d_labels()
    matches = {}
    for scene, pred in sorted(preds.items()):
        gt_file = os.path.join(bm.GT_TXT_DIR, scene + ".txt")
        arti = {np.int64(k): v for k, v in bm.load_gt_articulation(scene).items()}
        gt2pred, pred2gt = esi.assign_instances_for_scan(
            pred, gt_file, True, {"articulations_dict": arti}, False)
        matches[os.path.abspath(gt_file)] = {"gt": gt2pred, "pred": pred2gt,
                                             "scene": scene}
    return matches


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else (
        "/data/challenge/USDNet/results/eval_mov_val_einsum/debug/val_preds/preds.pkl")
    raw = pickle.load(open(src, "rb"))
    print(f"loaded {len(raw)} scenes from {src}", flush=True)

    pts_cache = {s: np.load(POINTS_DIR / f"{s}.npy", mmap_mode="r")[:, 0:3]
                 .astype(np.float64) for s in raw}

    def variant(axis_steps, origin_mode, qlo=2.0, qhi=98.0):
        out = {}
        for s, p in raw.items():
            ax = np.asarray(p["pred_axises"], np.float64)
            if "normalize" in axis_steps:
                ax = normalize(ax)
            if "canonical" in axis_steps:
                ax = snap_canonical(ax, p["pred_classes"])
            org = np.asarray(p["pred_origins"], np.float64)
            if origin_mode:
                org, _ = snap_origin(org, ax, p["pred_classes"], p["pred_masks"],
                                     pts_cache[s], origin_mode, qlo, qhi)
            out[s] = dict(p, pred_axises=ax, pred_origins=org)
        return out

    score(raw, "baseline")
    score(variant(["normalize"], None), "norm")
    score(variant(["normalize", "canonical"], None), "norm+canon")
    score(variant(["normalize", "canonical"], "pca"), "norm+canon+endpt(pca)")
    score(variant(["normalize", "canonical"], "perp"), "norm+canon+endpt(perp)")
    score(variant(["normalize", "canonical"], "near"), "norm+canon+nearest")
    score(variant(["normalize", "canonical"], "perp", 5.0, 95.0),
          "norm+canon+endpt(perp,5/95)")
    score(variant(["normalize", "canonical"], "perp", 0.0, 100.0),
          "norm+canon+endpt(perp,0/100)")


if __name__ == "__main__":
    main()
