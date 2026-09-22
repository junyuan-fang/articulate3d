#!/usr/bin/env python
"""Where does the ORIGIN criterion fail on the currently deployed pipeline?

Reframing the goal to 0.41 changes which factor matters. With
joint ~ AP50 x r_axis x r_origin and the test numbers 0.4370 / 93.85% / 82.42%:

    perfect axis, origin unchanged   -> 0.360   (not enough)
    perfect origin, axis unchanged   -> 0.410   (exactly the target)

So origin, at 82.4%, is the weak factor and the one worth attacking. The last
origin breakdown predates several changes, so re-measure.

The evaluator's origin test is stricter than it looks: BOTH projections must be
under 0.25 m -- one taken along the GT axis and one along the PREDICTED axis. A
correct origin can therefore fail purely because the axis is wrong, which means
some origin failures are really axis failures in disguise. This separates them.
"""
import os
import pickle
import sys

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
PKL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_leaf.pkl"
MASK_TH = 1e-3
ROT = 1
THRESH = 0.25


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def proj_dist(origin, gt_origin, axis):
    """Distance from the predicted origin to the GT axis line, as the evaluator
    computes it -- note it divides by norm(a), not norm(a)**2, so feed unit axes."""
    a = unit(np.asarray(axis, float))
    d = np.asarray(origin, float) - np.asarray(gt_origin, float)
    return float(np.linalg.norm(d - np.dot(d, a) * a))


preds = pickle.load(open(PKL, "rb"))
rows = []
for scene in sorted(preds):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    inst = np.asarray(a[:, 10]).astype(int)
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            g = h[k]
            if int(np.asarray(g["sem_id"])) == ROT:
                gt[int(k)] = (np.asarray(g["axis"], float),
                              np.asarray(g["origin"], float))
    if not gt:
        continue
    gmasks = {i: inst == i for i in gt}
    pm = np.asarray(preds[scene]["pred_masks"]) > MASK_TH
    sc = np.asarray(preds[scene]["pred_scores"]).ravel()
    ax = np.asarray(preds[scene]["pred_axises"], float)
    org = np.asarray(preds[scene]["pred_origins"], float)
    used = set()
    for qi in np.argsort(-sc):
        m = pm[:, qi]
        if not m.any():
            continue
        best, bid = 0.0, None
        for gi, gm in gmasks.items():
            if gi in used:
                continue
            iou = (m & gm).sum() / max((m | gm).sum(), 1)
            if iou > best:
                best, bid = iou, gi
        if best <= 0.5 or bid is None:
            continue
        used.add(bid)
        gax, gorg = gt[bid]
        d_gt = proj_dist(org[qi], gorg, gax)          # using the GT axis
        d_pr = proj_dist(org[qi], gorg, ax[qi])       # using our predicted axis
        axis_err = float(np.degrees(np.arccos(
            min(1.0, abs(float(np.dot(unit(ax[qi]), unit(gax))))))))
        rows.append(dict(scene=scene, d_gt=d_gt, d_pr=d_pr, axis_err=axis_err,
                         score=float(sc[qi])))

n = len(rows)
pass_gt = sum(r["d_gt"] < THRESH for r in rows)
pass_pr = sum(r["d_pr"] < THRESH for r in rows)
pass_both = sum(r["d_gt"] < THRESH and r["d_pr"] < THRESH for r in rows)
print(f"matched rotations: {n}\n")
print(f"origin passes using GT axis        : {pass_gt}/{n} = {100*pass_gt/n:.1f}%")
print(f"origin passes using PREDICTED axis : {pass_pr}/{n} = {100*pass_pr/n:.1f}%")
print(f"origin passes BOTH (what scores)   : {pass_both}/{n} = {100*pass_both/n:.1f}%")

killed = [r for r in rows if r["d_gt"] < THRESH and r["d_pr"] >= THRESH]
print(f"\ncorrect origin killed by a wrong axis: {len(killed)}")
if killed:
    e = np.array([r["axis_err"] for r in killed])
    print(f"  their axis errors: median {np.median(e):.1f} deg, "
          f"{int((e > 60).sum())} above 60 deg")

fail = [r for r in rows if r["d_gt"] >= THRESH]
print(f"\ngenuinely wrong origins (fail even with the GT axis): {len(fail)}")
if fail:
    d = np.array([r["d_gt"] for r in fail])
    print(f"  distance distribution: median {np.median(d):.3f} m")
    for lo, hi in [(0.25, 0.35), (0.35, 0.5), (0.5, 1.0), (1.0, 99)]:
        c = int(((d >= lo) & (d < hi)).sum())
        print(f"    [{lo:.2f}, {hi:.2f}) m : {c:3d}"
              + ("   <- near miss, a better side/endpoint choice reaches these"
                 if hi <= 0.35 else ""))
print(f"\nheadroom if every 'killed by axis' case were saved: "
      f"{100*(pass_both+len(killed))/n:.1f}% origin pass")
