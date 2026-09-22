#!/usr/bin/env python
"""Two untried geometric constraints for the axis deficit (-0.0075 vs linxii).

Rule A -- flat-lying rotation parts: a hinge axis lies IN the part's plane.
  If the panel normal is near-vertical (a lid lying flat), a vertical axis is
  geometrically impossible; the axis must be horizontal, along the long
  in-plane edge. Fires only on flat-lying rotation parts -> the 93% of
  vertical doors are untouched by construction.

Rule B -- handle-offset as a third translation-azimuth estimator: a drawer
  handle sticks out ALONG the slide direction, so (handle centroid - part
  centroid) flattened to the horizontal plane estimates the axis
  independently of both the model and the panel-normal PCA. Use it as an
  arbiter where model and PCA disagree (the 13.7% failure zone).
"""
import os
import pickle

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
RAW = "/data/challenge/USDNet/results/ev_v4init/debug/val_preds/preds.pkl"
INTER = "/data/challenge/USDNet/results/inference_inter_seg_val/debug/val_preds/preds.pkl"


def ang(a, b):
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


def pca_dirs(pts):
    c = pts - pts.mean(0)
    if len(c) > 4000:
        c = c[np.random.RandomState(0).choice(len(c), 4000, replace=False)]
    _, sv, vt = np.linalg.svd(c, full_matrices=False)
    return vt, sv


raw = pickle.load(open(RAW, "rb"))
inter = pickle.load(open(INTER, "rb"))

rows_A = []       # (fires, model_err, ruleA_err, gt_is_lateral)
rows_B = []       # (model_err, pca_err, handle_err, disagree)

for scene in sorted(raw):
    p = raw[scene]
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    pts_all = np.asarray(a[:, 0:3])
    inst = np.asarray(a[:, 10]).astype(int)
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            g = h[k]
            gt[int(k)] = (np.asarray(g["axis"], float), int(np.asarray(g["sem_id"])))
    if not gt:
        continue

    handles = []
    if scene in inter:
        ip = inter[scene]
        im = np.asarray(ip["pred_masks"]) > 1e-3
        isc = np.asarray(ip["pred_scores"]).ravel()
        for qi in np.where(isc > 0.5)[0]:
            sel = im[:, qi]
            if 10 <= sel.sum() < 20000:
                handles.append(pts_all[sel].mean(0))
    handles = np.array(handles) if handles else None

    gmasks = {i: inst == i for i in gt}
    pm = np.asarray(p["pred_masks"]) > 1e-3
    order = np.argsort(-np.asarray(p["pred_scores"]).ravel())
    used = set()
    for qi in order:
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
        gax, sem = gt[bid]
        pax = np.asarray(p["pred_axises"])[qi]
        part = pts_all[m]
        if len(part) < 10:
            continue
        vt, sv = pca_dirs(part)
        normal = vt[2]

        if sem == 1:
            # Rule A: fires when the panel normal is near-vertical
            fires = abs(normal[2]) > np.cos(np.deg2rad(30))
            ruleA_err = np.nan
            if fires:
                # long in-plane edge = first principal direction, flattened
                d = vt[0].copy()
                d[2] = 0.0
                n = np.linalg.norm(d)
                if n > 1e-3:
                    ruleA_err = ang(d / n, gax)
            gt_lateral = abs(gax[2] / max(np.linalg.norm(gax), 1e-9)) < 0.5
            rows_A.append((fires, ang(pax, gax), ruleA_err, gt_lateral))
        elif sem == 2:
            geo = None
            for v in (vt[2], vt[1]):
                f = v.copy(); f[2] = 0.0
                n = np.linalg.norm(f)
                if n > 0.3:
                    geo = f / n
                    break
            h_dir = None
            if handles is not None and len(handles):
                bbox_lo, bbox_hi = part.min(0) - 0.25, part.max(0) + 0.25
                near = handles[np.all((handles > bbox_lo) & (handles < bbox_hi), axis=1)]
                if len(near):
                    off = near[np.argmin(np.linalg.norm(near - part.mean(0), axis=1))] - part.mean(0)
                    off[2] = 0.0
                    n = np.linalg.norm(off)
                    if n > 0.02:
                        h_dir = off / n
            m_err = ang(pax, gax)
            p_err = ang(geo, gax) if geo is not None else np.nan
            h_err = ang(h_dir, gax) if h_dir is not None else np.nan
            dis = ang(pax, geo) if geo is not None else np.nan
            rows_B.append((m_err, p_err, h_err, dis))

A = rows_A
fires = [r for r in A if r[0]]
print(f"=== Rule A: flat-lying rotation parts (n_rot={len(A)}) ===")
print(f"  fires on {len(fires)} parts; of those GT-lateral: {sum(1 for r in fires if r[3])}")
for r in fires:
    print(f"    model_err {r[1]:5.1f}  ruleA_err {r[2]:5.1f}  gt_lateral={r[3]}")
verts_hit = [r for r in A if r[0] and not r[3]]
print(f"  false fires on vertical-GT parts: {len(verts_hit)}")

B = np.array(rows_B, float)
print(f"\n=== Rule B: handle-offset azimuth (n_tr={len(B)}) ===")
hb = B[~np.isnan(B[:, 2])]
print(f"  handle available: {len(hb)}/{len(B)}")
if len(hb):
    print(f"  errors: model med {np.median(hb[:,0]):.1f}  pca med {np.median(hb[:,1]):.1f}  handle med {np.median(hb[:,2]):.1f}")
    print(f"  <15deg: model {np.mean(hb[:,0]<15)*100:.0f}%  pca {np.mean(hb[:,1]<15)*100:.0f}%  handle {np.mean(hb[:,2]<15)*100:.0f}%")
# arbitration zone: model-PCA disagree > 25 deg
dz = B[(~np.isnan(B[:, 3])) & (B[:, 3] >= 25)]
print(f"  disagreement zone (model-vs-PCA >=25deg): {len(dz)} instances")
if len(dz):
    print(f"    model <15deg: {np.mean(dz[:,0]<15)*100:.0f}%   pca <15deg: {np.mean(dz[:,1]<15)*100:.0f}%")
    hdz = dz[~np.isnan(dz[:, 2])]
    print(f"    handle available in zone: {len(hdz)}; handle <15deg: "
          f"{np.mean(hdz[:,2]<15)*100:.0f}%" if len(hdz) else "    no handles in zone")
    if len(hdz):
        # arbiter: trust whichever of model/PCA agrees better with handle dir
        pick_pca = np.abs(hdz[:, 1] - hdz[:, 2]) < np.abs(hdz[:, 0] - hdz[:, 2])
        arb = np.where(pick_pca, hdz[:, 1], hdz[:, 0])
        print(f"    arbiter(pick closer to handle) <15deg: {np.mean(arb<15)*100:.0f}%")
