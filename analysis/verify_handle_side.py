#!/usr/bin/env python
"""Does "hinge is on the side opposite the handle" beat "endpoint nearest the
predicted origin" for choosing which end of a door carries the hinge?

Handles come from the *predicted* interactable masks (inter_trainval.ckpt val
inference) so the rule is deployable at test time. Evaluated on matched
rotation instances with the same endpoint construction as postprocess2.
"""
import os
import pickle

import h5py
import numpy as np

MOV = "/data/challenge/USDNet/results/ev_v4init/debug/val_preds/preds.pkl"
INTER = "/data/challenge/USDNet/results/inference_inter_seg_val/debug/val_preds/preds.pkl"
VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
MASK_TH = 1e-3
QLO, QHI = 5.0, 95.0
HANDLE_SCORE = 0.5


def endpoints(pts, axis):
    a = axis / max(np.linalg.norm(axis), 1e-9)
    c = pts.mean(0)
    X = pts - c
    Xp = X - np.outer(X @ a, a)
    _, _, vt = np.linalg.svd(Xp[np.random.RandomState(0).choice(len(Xp), min(len(Xp), 4000), replace=False)]
                             if len(Xp) > 4000 else Xp, full_matrices=False)
    d = vt[0] - (vt[0] @ a) * a
    d = d / max(np.linalg.norm(d), 1e-9)
    t = Xp @ d
    lo, hi = np.percentile(t, [QLO, QHI])
    return np.stack([c + lo * d, c + hi * d])


def perp_dist(p, o, a):
    a = a / max(np.linalg.norm(a), 1e-9)
    v = p - o
    return float(np.linalg.norm(v - (v @ a) * a))


mov = pickle.load(open(MOV, "rb"))
inter = pickle.load(open(INTER, "rb"))

n = agree = 0
right = {"origin": 0, "handle": 0, "handle_used": 0, "handle_avail": 0}
pass_ = {"origin": 0, "handle": 0}

for scene, p in sorted(mov.items()):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    pts_all = np.asarray(a[:, 0:3])
    inst = np.asarray(a[:, 10]).astype(int)
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            g = h[k]
            gt[int(k)] = (np.asarray(g["axis"], float),
                          np.asarray(g["origin"], float),
                          int(np.asarray(g["sem_id"])))
    if not gt:
        continue

    # predicted handle centroids for this scene
    handles = []
    if scene in inter:
        ip = inter[scene]
        im = np.asarray(ip["pred_masks"]) > MASK_TH
        isc = np.asarray(ip["pred_scores"]).ravel()
        for qi in np.where(isc > HANDLE_SCORE)[0]:
            sel = im[:, qi]
            if 10 <= sel.sum() < 20000:
                handles.append(pts_all[sel].mean(0))
    handles = np.array(handles) if handles else None

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
        gax, gor, sem = gt[bid]
        if sem != 1:
            continue
        part = pts_all[m]
        if len(part) < 3:
            continue
        pax = np.asarray(p["pred_axises"])[qi]
        ends = endpoints(part, pax)
        d0, d1 = (perp_dist(e, gor, gax) for e in ends)
        correct = int(d1 < d0)

        # rule A: nearest to predicted origin
        po = np.asarray(p["pred_origins"])[qi]
        pickA = int(np.linalg.norm(ends[1] - po) < np.linalg.norm(ends[0] - po))

        # rule B: farthest from the nearest handle (fallback to A)
        pickB, usedB = pickA, False
        if handles is not None:
            bbox_lo, bbox_hi = part.min(0) - 0.25, part.max(0) + 0.25
            near = handles[np.all((handles > bbox_lo) & (handles < bbox_hi), axis=1)]
            if len(near):
                right["handle_avail"] += 1
                dh0 = min(np.linalg.norm(near - ends[0], axis=1))
                dh1 = min(np.linalg.norm(near - ends[1], axis=1))
                pickB = int(dh1 > dh0)      # hinge = end farther from handle
                usedB = True

        n += 1
        agree += int(pickA == pickB)
        right["origin"] += int(pickA == correct)
        right["handle"] += int(pickB == correct)
        right["handle_used"] += int(usedB and pickB == correct)
        pass_["origin"] += int((d1 if pickA else d0) < 0.25)
        pass_["handle"] += int((d1 if pickB else d0) < 0.25)

print(f"matched rotations: {n}   handle available: {right['handle_avail']}")
print(f"pick agreement A vs B: {agree}/{n}")
print(f"correct-side rate:  origin-rule {right['origin']/n*100:5.1f}%   "
      f"handle-rule {right['handle']/n*100:5.1f}%")
print(f"origin<0.25 pass:   origin-rule {pass_['origin']/n*100:5.1f}%   "
      f"handle-rule {pass_['handle']/n*100:5.1f}%")
