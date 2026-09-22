#!/usr/bin/env python
"""Is there anything for a superpoint constraint to convert?

We need only +0.8% relative AP50 to take the last leaderboard sub-metric
(AP50_axis = AP50 x retention; at our current 93.85% retention, AP50 = 0.4406
clears the leader's 0.41346). Mask recall is 70.5%, so the headroom exists in
principle -- but only if the misses are NEAR misses.

Constraining masks to unions of superpoints cleans up ragged boundaries. That
can push an instance sitting just under IoU 0.5 over the line. It cannot
conjure an instance the model never localised at all.

So bucket every GT instance by the best IoU any prediction achieves against it.
Mass in 0.40-0.50 is convertible; mass near 0 is not. This decides whether to
spend an inference run.
"""
import os
import pickle
import sys

import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
PKL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_leaf.pkl"
MASK_TH = 1e-3
SCORE_TH = 0.0

preds = pickle.load(open(PKL, "rb"))
best_ious = []
for scene in sorted(preds):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    inst = np.asarray(a[:, 10]).astype(int)
    gt_ids = [i for i in np.unique(inst) if i >= 0]
    if not gt_ids:
        continue
    pm = np.asarray(preds[scene]["pred_masks"]) > MASK_TH
    sc = np.asarray(preds[scene]["pred_scores"]).ravel()
    keep = np.where(sc > SCORE_TH)[0]
    if len(keep) == 0:
        best_ious.extend([0.0] * len(gt_ids))
        continue
    pm = pm[:, keep]
    psum = pm.sum(0).astype(np.float64)
    for gi in gt_ids:
        gm = inst == gi
        g = gm.sum()
        if g < 10:
            continue
        inter = (gm[:, None] & pm).sum(0).astype(np.float64)
        iou = inter / np.maximum(psum + g - inter, 1)
        best_ious.append(float(iou.max()))

b = np.array(best_ious)
print(f"GT instances: {len(b)}\n")
edges = [0.0, 0.05, 0.2, 0.3, 0.4, 0.45, 0.5, 0.7, 1.01]
print(f"{'best IoU bucket':>18s} {'count':>7s} {'share':>7s}")
for lo, hi in zip(edges[:-1], edges[1:]):
    n = int(((b >= lo) & (b < hi)).sum())
    print(f"{f'[{lo:.2f}, {hi:.2f})':>18s} {n:7d} {100*n/len(b):6.1f}%")

matched = int((b >= 0.5).sum())
near = int(((b >= 0.4) & (b < 0.5)).sum())
nearer = int(((b >= 0.45) & (b < 0.5)).sum())
print(f"\nrecall @ IoU 0.5      : {100*matched/len(b):.1f}%  ({matched}/{len(b)})")
print(f"near misses [0.40,0.50): {near}  -> recall would be "
      f"{100*(matched+near)/len(b):.1f}% if all converted")
print(f"           [0.45,0.50): {nearer}")
print(f"hopeless   [0.00,0.20): {int((b < 0.2).sum())}")
print(f"\nAP50 needs +0.8% relative. Converting {near} of {len(b)} instances is "
      f"{100*near/len(b):.1f}% of GT -- compare against that.")
