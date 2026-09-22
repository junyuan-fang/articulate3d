#!/usr/bin/env python
"""Apply ONLY the sliding-leaf axis flip to an already post-processed preds pkl.

Why this exists: the leaf rule was first packaged on top of the v7 trainval
predictions, but the best test submission (0.3338) came from a different
checkpoint, whose pred_classes and pred_scores are not the same. Stacking the
two changes made the leaderboard delta unreadable. This script applies the flip
as a minimal edit to an existing package so the rule can be measured alone.

Safe to run on an already-geometrically-processed file:
  * only translation axes are touched, and the flip overrides whatever the
    normal/agreement-gate step chose for exactly the parts that have a sliding
    partner;
  * origins are left alone. Rotation origins were snapped perpendicular to
    their own axis, which this never changes, and the metric ignores
    translation origins entirely;
  * masks, classes and scores are copied through untouched, so AP50 is
    unchanged by construction.

A second minimal edit rides along: force every ROTATION axis to vertical. The
axis head has collapsed onto gravity so completely that all but 4.4% of its
rotation predictions are already exactly [0,0,1], and printing the survivors
against ground truth shows they are simply wrong -- the one non-vertical
prediction among matched validation instances has a vertical GT. Snapping the
rest costs nothing and is parameter-free:

    AP50_axis  0.5046 -> 0.5063     AP50_axis_origin  0.4605 -> 0.4621

This does not touch the eight lateral hinges, which are already predicted
vertical and already failing; nothing here can fix those.

Usage:  python leaf_flip_only.py IN.pkl OUT.pkl POINTS_DIR [--no-rotz]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from postprocess2 import (MASK_THRESH, ROTATION, TRANSLATION, _is_sliding_leaf,
                          _panel, PLANAR_TH)

Z = np.array([0.0, 0.0, 1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("points_dir")
    ap.add_argument("--no-rotz", action="store_true",
                    help="skip forcing rotation axes vertical")
    args = ap.parse_args()

    preds = pickle.load(open(args.inp, "rb"))
    pdir = Path(args.points_dir)
    flipped = 0
    snapped = 0
    shortest = float("inf")
    for scene, p in preds.items():
        pts = np.load(pdir / f"{scene}.npy", mmap_mode="r")[:, 0:3].astype(np.float64)
        masks = np.asarray(p["pred_masks"])
        if pts.shape[0] != masks.shape[0]:
            raise SystemExit(f"{scene}: point/mask row mismatch -- wrong points_dir?")
        classes = np.asarray(p["pred_classes"]).ravel()
        axes = np.asarray(p["pred_axises"], dtype=np.float64).copy()

        panels = {}
        for i in range(axes.shape[0]):
            if classes[i] != TRANSLATION:
                continue
            sel = masks[:, i] > MASK_THRESH
            if sel.sum() < 10:
                continue
            q = _panel(pts[sel])
            if q is not None:
                panels[i] = q

        for i, q in panels.items():
            if q["planarity"] >= PLANAR_TH:
                continue
            if _is_sliding_leaf(q, [v for j, v in panels.items() if j != i]):
                axes[i] = q["inp"]
                flipped += 1
                shortest = min(shortest, q["h"])

        if not args.no_rotz:
            rot = classes == ROTATION
            if rot.any():
                u = axes[rot] / np.maximum(np.linalg.norm(axes[rot], axis=1,
                                                          keepdims=True), 1e-9)
                snapped += int((np.abs(u[:, 2]) < 0.999).sum())
                axes[rot] = Z

        p["pred_axises"] = axes

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(preds, open(args.out, "wb"))
    s = f"{shortest:.2f} m" if flipped else "n/a"
    print(f"{len(preds)} scenes | leaves flipped: {flipped} (shortest {s}) | "
          f"rotation axes snapped to vertical: {snapped}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
