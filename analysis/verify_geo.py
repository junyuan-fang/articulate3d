#!/usr/bin/env python
"""Test two training-free geometric hypotheses against val GT.

Reverse-engineered from linxii's leaderboard numbers (docs/03): their axis
retention is 95% vs our 89% with segmentation quality equal to ours, which a
query-regression head + canonical snapping cannot reach -- but part *geometry*
might:

H-A (translation): the slide axis of a drawer/door is the normal of its front
    panel, i.e. the smallest-extent PCA direction of the part's points,
    flattened into the horizontal plane. Compare against the model's azimuth.

H-B (rotation): lateral hinges (oven/dishwasher doors, lids) live on wide-flat
    parts. Test how well "horizontal extent much larger than vertical extent"
    predicts a horizontal GT axis.

Uses the raw bin-head predictions (ev_v4init) and GT; greedy IoU>0.5 matching,
same convention as the benchmark.
"""
import glob
import os
import pickle

import h5py
import numpy as np

PRED = "/data/challenge/USDNet/results/ev_v4init/debug/val_preds/preds.pkl"
VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
MASK_TH = 1e-3


def ang_deg(a, b):
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


def pca_axis_horizontal(pts):
    """Smallest-extent PCA direction, flattened to the horizontal plane."""
    c = pts - pts.mean(0)
    # subsample for speed; direction estimate is robust to it
    if len(c) > 4000:
        c = c[np.random.RandomState(0).choice(len(c), 4000, replace=False)]
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    for v in (vt[2], vt[1]):          # smallest first; fall back if vertical
        f = v.copy()
        f[2] = 0.0
        n = np.linalg.norm(f)
        if n > 0.3:                    # direction has a usable azimuth
            return f / n
    return None


preds = pickle.load(open(PRED, "rb"))
rows_t, rows_r = [], []

for scene, p in sorted(preds.items()):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    pts = np.asarray(a[:, 0:3])
    inst = np.asarray(a[:, 10]).astype(int)
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            g = h[k]
            gt[int(k)] = (np.asarray(g["axis"], float), int(np.asarray(g["sem_id"])))
    if not gt:
        continue
    gt_masks = {i: inst == i for i in gt}
    pm = np.asarray(p["pred_masks"]) > MASK_TH
    order = np.argsort(-np.asarray(p["pred_scores"]))
    used = set()
    for qi in order:
        m = pm[:, qi]
        if not m.any():
            continue
        best, bid = 0.0, None
        for gi, gm in gt_masks.items():
            if gi in used:
                continue
            iou = (m & gm).sum() / max((m | gm).sum(), 1)
            if iou > best:
                best, bid = iou, gi
        if best <= 0.5 or bid is None:
            continue
        used.add(bid)
        gax, sem = gt[bid]
        pax = np.asarray(p["pred_axises"])[qi]
        part = pts[m]
        if sem == 2:                                   # translation: H-A
            geo = pca_axis_horizontal(part)
            rows_t.append((ang_deg(pax, gax),
                           ang_deg(geo, gax) if geo is not None else np.nan))
        elif sem == 1:                                 # rotation: H-B
            ext = part.max(0) - part.min(0)
            horiz_ext = float(max(ext[0], ext[1]))
            vert_ext = float(ext[2])
            gt_vertical = abs(gax[2] / max(np.linalg.norm(gax), 1e-9)) > 0.5
            rows_r.append((horiz_ext, vert_ext, gt_vertical, ang_deg(pax, gax)))

t = np.array(rows_t, float)
print(f"=== H-A translation (n={len(t)}) ===")
ok = ~np.isnan(t[:, 1])
print(f"  PCA azimuth usable: {ok.sum()}/{len(t)}")
m, g = t[ok, 0], t[ok, 1]
for name, x in (("model", m), ("PCA  ", g)):
    print(f"  {name}: median {np.median(x):5.2f}  p75 {np.percentile(x,75):5.2f}  "
          f"<15deg {np.mean(x<15)*100:5.1f}%")
hyb = np.where(g < m, g, m)
print(f"  oracle-best-of-two:      <15deg {np.mean(hyb<15)*100:5.1f}%")
# realistic hybrid: trust PCA only when the two agree loosely, else keep model
agree = np.abs(m - g) < 25
hyb2 = np.where(agree & (g < m), g, m)
print(f"  PCA-when-close hybrid:   <15deg {np.mean(hyb2<15)*100:5.1f}%")

r = np.array(rows_r, float)
print(f"\n=== H-B rotation (n={len(r)}) ===")
gtv = r[:, 2].astype(bool)
print(f"  GT vertical {gtv.sum()} / lateral {(~gtv).sum()}")
for k in (1.5, 2.0, 3.0):
    pred_lat = r[:, 0] > k * r[:, 1]          # wide-flat => lateral
    tp = (pred_lat & ~gtv).sum()
    fp = (pred_lat & gtv).sum()
    fn = ((~pred_lat) & ~gtv).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"  horiz>{k}x vert: pred_lateral={pred_lat.sum()}  "
          f"precision {prec*100:5.1f}%  recall {rec*100:5.1f}%")
lat = r[~gtv]
if len(lat):
    print("  lateral parts (horiz_ext, vert_ext, model_axis_err):")
    for h_, v_, _, e_ in lat:
        print(f"    {h_:5.2f} x {v_:5.2f}  err {e_:5.1f} deg")
