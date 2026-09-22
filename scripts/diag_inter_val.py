#!/usr/bin/env python
"""Failure-mode diagnosis of the baseline inter model on val.

Buckets every GT interaction instance by the best IoU any prediction achieves
(same logic as analysis/gate_superpoint_headroom.py but for the inter task).
Mass in [0.40, 0.50) = near misses convertible by boundary cleanup; mass near 0
= detection failures only more recall can fix.
Columns in the processed inter npy: 0-2 xyz, 3-5 rgb, 6-8 normal,
9 semantic (0=bg, filtered), 10 instance, 12 inter_id.
"""
import os, pickle, sys
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_inter/validation"
PKL = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/challenge/USDNet/results/inference_inter_seg_val/debug/val_preds/preds.pkl"
MASK_TH = 1e-3

preds = pickle.load(open(PKL, "rb"))
best_ious, sizes, n_pred_tot = [], [], 0
for scene in sorted(preds):
    p = os.path.join(VAL, scene + ".npy")
    if not os.path.exists(p):
        continue
    a = np.load(p, mmap_mode="r")
    sem = np.asarray(a[:, 9]).astype(int)
    inst = np.asarray(a[:, 10]).astype(int)
    pm = np.asarray(preds[scene]["pred_masks"]) > MASK_TH
    n_pred_tot += pm.shape[1]
    psum = pm.sum(0).astype(np.float64)
    for gi in np.unique(inst):
        gm = inst == gi
        # skip background "instance": majority label 0
        lab = np.bincount(sem[gm]).argmax()
        if lab == 0:
            continue
        g = gm.sum()
        if g < 10:
            continue
        inter = (gm[:, None] & pm).sum(0).astype(np.float64)
        iou = inter / np.maximum(psum + g - inter, 1)
        best_ious.append(float(iou.max()))
        sizes.append(int(g))

b, s = np.array(best_ious), np.array(sizes)
print(f"scenes: {len(preds)}  GT instances: {len(b)}  preds total: {n_pred_tot}")
print(f"recall@0.5 (oracle upper bound incl. low-score preds): {(b>=0.5).mean():.3f}")
edges = [0.0, 0.05, 0.2, 0.3, 0.4, 0.45, 0.5, 0.7, 1.01]
print(f"{'best IoU bucket':>16s} {'count':>6s} {'share':>7s} {'med size':>9s}")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (b >= lo) & (b < hi)
    n = int(m.sum())
    med = int(np.median(s[m])) if n else 0
    print(f"{f'[{lo:.2f},{hi:.2f})':>16s} {n:6d} {100*n/len(b):6.1f}% {med:9d}")
print("\nGT size overall: median", int(np.median(s)), " p25", int(np.percentile(s,25)),
      " p75", int(np.percentile(s,75)))
