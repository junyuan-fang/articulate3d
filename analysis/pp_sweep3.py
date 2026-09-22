#!/usr/bin/env python
"""What does the axis bin+residual head do at initialisation?

Both of its output layers are zero-initialised and its logits are biased by
|cos(anchor, free_axis)|, so at load time it emits, for every query, the anchor
nearest to the free-regression axis it replaces -- unconditionally, with no
angle threshold. That transform can be applied to a baseline preds.pkl on the
CPU, which prices the warm start before spending five GPU-hours on it.

Rows:
  anchor5-hard     the head's exact init behaviour (5 anchors: z + 4 azimuths)
  anchor5-thresh   same but only when already within THRESH of an anchor
  anchor9-hard     finer azimuth grid (8 azimuths), to size the bin resolution
  class-canon      the verified post-processing rule (rotation->z, translation
                   ->horizontal, only within 30 deg) for reference

Usage: python pp_sweep3.py [preds.pkl]
"""
import sys
import pickle

import numpy as np

from pp_sweep import normalize, snap_canonical, score

S = 2.0 ** -0.5


def anchors(n_azimuth):
    a = [[0.0, 0.0, 1.0]]
    for k in range(n_azimuth):
        t = np.pi * k / n_azimuth          # |cos| is sign-agnostic: 180 deg spans all
        a.append([np.cos(t), np.sin(t), 0.0])
    return np.asarray(a)


def snap_anchor(axes, A, thresh_deg=None):
    ax = normalize(np.asarray(axes, np.float64))
    cos = np.abs(ax @ A.T)                 # (n, A)
    k = cos.argmax(1)
    out = A[k].copy()
    if thresh_deg is not None:
        keep = cos.max(1) >= np.cos(np.deg2rad(thresh_deg))
        out[~keep] = ax[~keep]
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else (
        "/data/challenge/USDNet/results/eval_mov_val_einsum/debug/val_preds/preds.pkl")
    raw = pickle.load(open(src, "rb"))
    print(f"loaded {len(raw)} scenes", flush=True)

    def variant(fn):
        return {s: dict(p, pred_axises=fn(p["pred_axises"], p["pred_classes"]))
                for s, p in raw.items()}

    def vh(axes, classes=None, thresh_deg=None):
        """Two data-dependent anchors: gravity, or the axis's own horizontal
        projection. Unlike a fixed azimuth grid this keeps azimuth resolution
        exactly, which is what translation axes need (they are horizontal but
        their azimuth is arbitrary)."""
        ax = normalize(np.asarray(axes, np.float64))
        cz = np.abs(ax[:, 2])
        flat = ax.copy()
        flat[:, 2] = 0.0
        n = np.linalg.norm(flat, axis=1, keepdims=True)
        flat = np.where(n > 1e-6, flat / np.maximum(n, 1e-9), ax)
        vert = cz >= (S if thresh_deg is None else np.cos(np.deg2rad(thresh_deg)))
        horiz = cz <= (S if thresh_deg is None
                       else np.cos(np.deg2rad(90 - thresh_deg)))
        out = ax.copy()
        out[vert] = [0.0, 0.0, 1.0]
        out[horiz] = flat[horiz]
        return out

    A5, A9 = anchors(4), anchors(8)
    score(variant(lambda a, c: normalize(np.asarray(a, np.float64))), "norm only")
    score(variant(lambda a, c: snap_canonical(a, c, 30.0)), "class-canon(30) [verified rule]")
    score(variant(lambda a, c: snap_anchor(a, A5)), "anchor5-hard [head init]")
    score(variant(lambda a, c: snap_anchor(a, A5, 22.5)), "anchor5-thresh(22.5)")
    score(variant(lambda a, c: snap_anchor(a, A9)), "anchor9-hard")
    score(variant(lambda a, c: snap_anchor(a, A9, 11.25)), "anchor9-thresh(11.25)")
    score(variant(lambda a, c: vh(a)), "vert-or-horiz hard (45 deg split)")
    score(variant(lambda a, c: vh(a, thresh_deg=30.0)), "vert-or-horiz(30) class-free")
    score(variant(lambda a, c: snap_canonical(a, c, 45.0)), "class-canon(45)")


if __name__ == "__main__":
    main()
