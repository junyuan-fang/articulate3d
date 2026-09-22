#!/usr/bin/env python
"""Training-free post-processing for USDNet articulation predictions.

Two interventions, both derived from the val diagnosis (docs/04-diagnosis.md).
Neither touches pred_masks, so AP50 is unchanged by construction.

A. Gravity snap (rotation only). The axis head has collapsed onto the gravity
   prior: 93.3% of matched rotation GT axes are within 15 deg of vertical, and
   the trivial "always predict z" baseline actually beats the model on rotation
   axes (93.3% vs 88.3%, 0 wins / 9 losses). Snapping near-vertical rotation
   axes to exactly z removes the residual jitter around the dominant mode.
   Cost: it also locks in the 8 horizontal-hinge failures -- but those are
   already 8/8 wrong, so nothing is lost.

D. Origin snap. Origin error is an absolute-scale problem uncorrelated with
   part size (Spearman 0.006), and predicted origins frequently sit off the
   part entirely. Snapping each origin to the nearest point of its own
   predicted mask pulls it back onto the object. No hyperparameter.

Measured on val (41 scenes): AP50_axis_origin 0.3214 -> 0.3718 (+0.0505),
AP50 unchanged at 0.5711.

Usage:
    python postprocess.py IN.pkl OUT.pkl POINTS_DIR [--no-gravity] [--no-origin]

POINTS_DIR holds the <scene>.npy point clouds matching the predictions
(cols 0-2 are xyz), e.g. .../articulate3d_challenge_mov/test
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

GRAVITY = np.array([0.0, 0.0, 1.0])
GRAVITY_SNAP_DEG = 30.0   # plateau on val; 30 and 45 give the same AP
ROTATION_CLASS = 1
MASK_THRESH = 1e-3        # ">1e-3 = point in instance" per the submission spec


def snap_axes_to_gravity(axes, classes, thresh_deg=GRAVITY_SNAP_DEG):
    """Snap near-vertical rotation axes to exactly +z. Returns (axes, n_snapped)."""
    axes = np.asarray(axes, dtype=np.float64).copy()
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    unit = axes / np.maximum(norms, 1e-9)
    # |cos| : the metric is sign-agnostic, so a downward axis is still vertical
    cos = np.abs(unit @ GRAVITY)
    is_rot = np.asarray(classes).ravel() == ROTATION_CLASS
    hit = is_rot & (cos >= np.cos(np.deg2rad(thresh_deg)))
    axes[hit] = GRAVITY * np.maximum(norms[hit], 1e-9)   # keep original magnitude
    return axes, int(hit.sum())


def snap_origins_to_mask(origins, masks, points):
    """Move each origin to the nearest point of its own predicted mask.

    Instances with an empty mask are left untouched -- there is nothing to snap
    to, and they cannot be matched by the evaluator anyway.
    """
    origins = np.asarray(origins, dtype=np.float64).copy()
    masks = np.asarray(masks)
    n_snapped = 0
    for i in range(origins.shape[0]):
        sel = masks[:, i] > MASK_THRESH
        if not sel.any():
            continue
        pts = points[sel]
        d = np.linalg.norm(pts - origins[i], axis=1)
        origins[i] = pts[int(np.argmin(d))]
        n_snapped += 1
    return origins, n_snapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("points_dir")
    ap.add_argument("--no-gravity", action="store_true")
    ap.add_argument("--no-origin", action="store_true")
    ap.add_argument("--gravity-deg", type=float, default=GRAVITY_SNAP_DEG)
    args = ap.parse_args()

    preds = pickle.load(open(args.inp, "rb"))
    pdir = Path(args.points_dir)
    tot_ax = tot_or = 0

    for scene, p in preds.items():
        if not args.no_gravity:
            p["pred_axises"], n = snap_axes_to_gravity(
                p["pred_axises"], p["pred_classes"], args.gravity_deg)
            tot_ax += n
        if not args.no_origin:
            pts = np.load(pdir / f"{scene}.npy", mmap_mode="r")[:, 0:3].astype(np.float64)
            if pts.shape[0] != np.asarray(p["pred_masks"]).shape[0]:
                raise SystemExit(
                    f"{scene}: point count {pts.shape[0]} != mask rows "
                    f"{np.asarray(p['pred_masks']).shape[0]} -- wrong points_dir?")
            p["pred_origins"], n = snap_origins_to_mask(
                p["pred_origins"], p["pred_masks"], pts)
            tot_or += n

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(preds, f)
    print(f"{len(preds)} scenes | axes snapped to gravity: {tot_ax} | "
          f"origins snapped to mask: {tot_or}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
