#!/usr/bin/env python
"""Snap predicted instance masks to superpoint boundaries.

This is the one intervention so far that changes `pred_masks`, so unlike every
other rule in the stack it CAN move AP50 -- in either direction. That is the
point: AP50_axis = AP50 x retention, and at our current 93.85% retention only
+0.8% relative AP50 is needed to pass the leaderboard's axis sub-metric, which
is the last one we trail. gate_superpoint_headroom.py found 24 validation
instances stuck at IoU [0.40, 0.50), which a boundary cleanup could convert.

For each predicted mask and each superpoint, the whole superpoint is included
when more than VOTE of its points are in the mask, and dropped otherwise. Parts
are contiguous geometric regions, so a mask that disagrees with the geometry
inside one small patch is usually the mask being wrong.

Everything except pred_masks is copied through untouched, but note that unlike
leaf_flip_only.py this does NOT leave the other metrics invariant: origins and
axes are computed downstream from the masks, so re-run the geometry stack after
this, not before.

Usage:
  python snap_masks_to_superpoints.py IN.pkl OUT.pkl SUPERPOINT_DIR [--vote 0.5]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("sp_dir")
    ap.add_argument("--vote", type=float, default=0.5)
    ap.add_argument("--mask-thresh", type=float, default=1e-3)
    ap.add_argument("--conservative", nargs=2, type=float, metavar=("LO", "HI"),
                    default=None,
                    help="instead of rebuilding the mask from superpoints, only "
                         "fill superpoints above HI coverage and drop those below "
                         "LO, leaving the rest of the mask as predicted")
    args = ap.parse_args()

    preds = pickle.load(open(args.inp, "rb"))
    spd = Path(args.sp_dir)
    changed = total = 0
    for scene, p in preds.items():
        f = spd / f"{scene}.npy"
        if not f.exists():
            raise SystemExit(f"no superpoints for {scene} at {f}")
        seg = np.load(f)
        m = np.asarray(p["pred_masks"])
        if m.shape[0] != len(seg):
            raise SystemExit(f"{scene}: {m.shape[0]} mask rows vs {len(seg)} superpoint rows")
        binm = m > args.mask_thresh

        nseg = int(seg.max()) + 1
        counts = np.bincount(seg, minlength=nseg).astype(np.float64)
        # per (segment, query) count of in-mask points, via a sparse-free bincount
        out = np.zeros_like(binm)
        for q in range(binm.shape[1]):
            col = binm[:, q]
            if not col.any():
                continue
            hit = np.bincount(seg[col], minlength=nseg).astype(np.float64)
            frac = hit / np.maximum(counts, 1)
            if args.conservative:
                lo, hi = args.conservative
                newcol = col.copy()
                newcol[(frac > hi)[seg]] = True      # fill nearly-covered patches
                newcol[(frac < lo)[seg]] = False     # drop nearly-empty ones
            else:
                newcol = (frac > args.vote)[seg]
            if not np.array_equal(newcol, col):
                changed += 1
            out[:, q] = newcol
            total += 1
        # preserve dtype/semantics of the original mask array
        p["pred_masks"] = out.astype(m.dtype) if m.dtype != bool else out

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(preds, open(args.out, "wb"))
    print(f"{len(preds)} scenes | masks altered: {changed}/{total} "
          f"({100*changed/max(total,1):.1f}%) | vote={args.vote}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
