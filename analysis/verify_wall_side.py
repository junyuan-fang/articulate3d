#!/usr/bin/env python
"""Side-choice arbiter candidate: the hinge edge abuts the frame/wall.

For each rotation part, look OUTWARD past each endpoint (along the widest
perpendicular-to-axis direction) and count scene points that do NOT belong to
the part. The hinge side should show dense static geometry (frame/wall); the
handle side opens into a gap. Evaluated on matched val rotations against the
current rules (nearest-to-predicted-origin, handle-opposite).
"""
import os
import pickle

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
RAW = "/data/challenge/USDNet/results/ev_v4init/debug/val_preds/preds.pkl"
INTER = "/data/challenge/USDNet/results/inference_inter_seg_val/debug/val_preds/preds.pkl"
QLO, QHI = 5.0, 95.0


def perp_dist(p, o, a):
    a = a / max(np.linalg.norm(a), 1e-9)
    v = p - o
    return float(np.linalg.norm(v - (v @ a) * a))


raw = pickle.load(open(RAW, "rb"))
inter = pickle.load(open(INTER, "rb"))

n = 0
acc = {"origin": 0, "handle": 0, "wall": 0, "handle+wall": 0}
pas = {k: 0 for k in acc}

for scene in sorted(raw):
    p = raw[scene]
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    pts_all = np.asarray(a[:, 0:3])
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            g = h[k]
            gt[int(k)] = (np.asarray(g["axis"], float), np.asarray(g["origin"], float),
                          int(np.asarray(g["sem_id"])))
    if not gt:
        continue
    inst = np.asarray(a[:, 10]).astype(int)
    gmasks = {i: inst == i for i in gt}

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
        gax, gor, sem = gt[bid]
        if sem != 1:
            continue
        part = pts_all[m]
        if len(part) < 10:
            continue
        pax = np.asarray(p["pred_axises"])[qi]
        ax = pax / max(np.linalg.norm(pax), 1e-9)
        c = part.mean(0)
        X = part - c
        Xp = X - np.outer(X @ ax, ax)
        sub = Xp if len(Xp) <= 4000 else Xp[np.random.RandomState(0).choice(len(Xp), 4000, replace=False)]
        _, _, vt = np.linalg.svd(sub, full_matrices=False)
        d = vt[0] - (vt[0] @ ax) * ax
        d = d / max(np.linalg.norm(d), 1e-9)
        t = Xp @ d
        lo, hi = np.percentile(t, [QLO, QHI])
        ends = np.stack([c + lo * d, c + hi * d])

        d0, d1 = (perp_dist(e, gor, gax) for e in ends)
        correct = int(d1 < d0)

        po = np.asarray(p["pred_origins"])[qi]
        pick_o = int(np.linalg.norm(ends[1] - po) < np.linalg.norm(ends[0] - po))

        pick_h, has_h = pick_o, False
        if handles is not None and len(handles):
            blo, bhi = part.min(0) - 0.25, part.max(0) + 0.25
            near = handles[np.all((handles > blo) & (handles < bhi), axis=1)]
            if len(near):
                dh0 = np.linalg.norm(near - ends[0], axis=1).min()
                dh1 = np.linalg.norm(near - ends[1], axis=1).min()
                pick_h, has_h = int(dh1 > dh0), True

        # wall rule: static-point density in a probe box beyond each endpoint
        other = ~m
        static = pts_all[other]
        # coarse pre-crop for speed
        blo, bhi = part.min(0) - 0.6, part.max(0) + 0.6
        near_static = static[np.all((static > blo) & (static < bhi), axis=1)]
        dens = []
        for k_, e in enumerate(ends):
            sgn = -1.0 if k_ == 0 else 1.0
            v = near_static - e
            along = v @ (sgn * d)                       # outward
            across = np.linalg.norm(v - np.outer(v @ d, d), axis=1)
            cnt = int(((along > 0.02) & (along < 0.25) & (across < 0.25)).sum())
            dens.append(cnt)
        pick_w = int(dens[1] > dens[0])                 # hinge = denser outward side

        pick_hw = pick_h if has_h else pick_w

        n += 1
        for name, pk in (("origin", pick_o), ("handle", pick_h),
                         ("wall", pick_w), ("handle+wall", pick_hw)):
            acc[name] += int(pk == correct)
            pas[name] += int((d1 if pk else d0) < 0.25)

print(f"matched rotations: {n}")
for k in acc:
    print(f"  {k:12s} side-acc {acc[k]/n*100:5.1f}%   origin<0.25 pass {pas[k]/n*100:5.1f}%")
