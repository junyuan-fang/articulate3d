#!/usr/bin/env python
"""Decisive check for flip-TTA: does the mirrored run match any GT instance the
original run misses? If the union recall barely exceeds the single-run recall,
merging has no ceiling and the concat/NMS losses are unrecoverable."""
import os
import pickle

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
A = pickle.load(open("/data/challenge/USDNet/results/ev_v4init/debug/val_preds/preds.pkl", "rb"))
B = pickle.load(open("/data/challenge/USDNet/results/ev_flipx/debug/val_preds/preds.pkl", "rb"))


def hits(masks, gt_masks):
    got = set()
    sums = masks.sum(0)
    for gi, gm in gt_masks.items():
        g = gm.sum()
        inter = (masks[gm]).sum(0)          # (Q,) intersection counts
        union = sums + g - inter
        iou = np.where(union > 0, inter / union, 0)
        if (iou > 0.5).any():
            got.add(gi)
    return got


tot = onlyA = onlyB = both = neither = 0
for scene in sorted(A):
    a = np.asarray(A[scene]["pred_masks"]) > 1e-3
    b = np.asarray(B[scene]["pred_masks"]) > 1e-3
    inst = np.asarray(np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")[:, 10]).astype(int)
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        ids = [int(k) for k in h.keys()]
    gm = {i: inst == i for i in ids}
    ha, hb = hits(a, gm), hits(b, gm)
    tot += len(gm)
    both += len(ha & hb)
    onlyA += len(ha - hb)
    onlyB += len(hb - ha)
    neither += len(gm) - len(ha | hb)

print(f"GT total {tot}")
print(f"  both hit: {both}   original-only: {onlyA}   mirrored-only: {onlyB}   missed by both: {neither}")
print(f"  recall  original {(both+onlyA)/tot*100:.1f}%   mirrored {(both+onlyB)/tot*100:.1f}%   union {(tot-neither)/tot*100:.1f}%")
