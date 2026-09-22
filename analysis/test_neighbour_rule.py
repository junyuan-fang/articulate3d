#!/usr/bin/env python
"""Does the arrangement of neighbouring panels say slide-sideways vs slide-out?

scan_slider_rule.py ruled out the part's own shape: at planarity < 0.05 the two
classes interleave completely between 0.5 and 0.85 m of height, because a flat
0.6 x 0.6 m panel really can be either a wardrobe leaf or a large drawer front.

The remaining signal is relational, and it has a clear physical story:

  * a stack of drawers shares ONE plane and its parts are separated VERTICALLY
    (offset along the panel normal ~ 0, offset in z large);
  * a set of sliding leaves sits in SLIGHTLY DIFFERENT planes, one leaf passing
    behind the next, and its parts are separated HORIZONTALLY within the plane
    (offset along the normal ~ one leaf thickness, offset in z ~ 0).

So for each planar translation part, find its parallel-plane neighbours among
the other predicted translation instances and measure that offset geometry.
Report the two groups separately; adopt nothing unless they separate widely.
"""
import os
import pickle

import h5py
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
RAW = "/data/challenge/USDNet/results/ev_v4init/debug/val_preds/preds.pkl"
MASK_TH = 1e-3
TRANSLATION = 2
SCORE_TH = 0.3
PARALLEL_DEG = 20.0


def ang(a, b):
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


def flatten(v):
    f = np.asarray(v, float).copy()
    f[2] = 0.0
    n = np.linalg.norm(f)
    return f / n if n > 1e-6 else None


def panel(pts):
    c = pts - pts.mean(0)
    s = c if len(c) <= 4000 else c[np.random.RandomState(0).choice(len(c), 4000, False)]
    _, sv, vt = np.linalg.svd(s, full_matrices=False)
    nrm = flatten(vt[2])
    if nrm is None:
        return None
    return dict(nrm=nrm, inp=np.array([-nrm[1], nrm[0], 0.0]),
                pl=float(sv[2] / max(sv[1], 1e-9)), cen=pts.mean(0),
                h=float(np.ptp(pts[:, 2])), w=float(np.ptp(c @ np.array(
                    [-nrm[1], nrm[0], 0.0]))))


def neighbour_feats(me, others):
    """Best evidence that `me` belongs to a set of side-by-side sliding leaves."""
    best = None
    for o in others:
        if ang(me["nrm"], o["nrm"]) > PARALLEL_DEG:
            continue
        d = o["cen"] - me["cen"]
        dn = abs(float(d @ me["nrm"]))          # separation through the plane
        di = abs(float(d @ me["inp"]))          # separation across the plane
        dz = abs(float(d[2]))                   # separation in height
        # leaves overlap in height and sit beside each other; drawers do not
        side_by_side = di > 0.5 * (me["w"] + o["w"]) * 0.4 and dz < 0.5 * min(me["h"], o["h"])
        cand = dict(dn=dn, di=di, dz=dz, side=side_by_side)
        if best is None or (cand["side"] and not best["side"]) or \
           (cand["side"] == best["side"] and di > best["di"]):
            best = cand
    return best


raw = pickle.load(open(RAW, "rb"))
rows = []
for scene in sorted(raw):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    pts_all = np.asarray(a[:, 0:3])
    inst = np.asarray(a[:, 10]).astype(int)
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            g = h[k]
            if int(np.asarray(g["sem_id"])) == 2:
                gt[int(k)] = np.asarray(g["axis"], float)
    if not gt:
        continue
    pm = np.asarray(raw[scene]["pred_masks"]) > MASK_TH
    sc = np.asarray(raw[scene]["pred_scores"]).ravel()
    cl = np.asarray(raw[scene]["pred_classes"]).ravel()

    # every predicted translation panel in the scene, for the neighbour search
    panels = {}
    for qi in np.where((cl == TRANSLATION) & (sc > SCORE_TH))[0]:
        sel = pm[:, qi]
        if sel.sum() < 50:
            continue
        p = panel(pts_all[sel])
        if p is not None:
            panels[int(qi)] = p

    gmasks = {i: inst == i for i in gt}
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
        if m.sum() < 10:
            continue
        p = panel(pts_all[m])
        if p is None:
            continue
        nb = neighbour_feats(p, [v for k, v in panels.items() if k != int(qi)])
        rows.append(dict(scene=scene, pl=p["pl"], h=p["h"], w=p["w"], nb=nb,
                         gn=ang(gt[bid], p["nrm"]), gi=ang(gt[bid], p["inp"])))

sub = [r for r in rows if r["pl"] < 0.05]
print(f"planar parts: {len(sub)}\n")
print(f"{'label':>8s} {'dn':>6s} {'di':>6s} {'dz':>6s} {'side':>5s} {'h':>5s} scene")
for r in sorted(sub, key=lambda x: (x["gi"] >= x["gn"], -x["h"])):
    lab = "INPLANE" if r["gi"] < r["gn"] else "normal"
    nb = r["nb"]
    if nb is None:
        print(f"{lab:>8s} {'--':>6s} {'--':>6s} {'--':>6s} {'--':>5s} "
              f"{r['h']:5.2f} {r['scene']}")
    else:
        print(f"{lab:>8s} {nb['dn']:6.3f} {nb['di']:6.2f} {nb['dz']:6.2f} "
              f"{str(nb['side']):>5s} {r['h']:5.2f} {r['scene']}")

pos = [r for r in sub if r["gi"] < r["gn"]]
neg = [r for r in sub if r["gi"] >= r["gn"]]
for name, grp in (("INPLANE", pos), ("normal", neg)):
    has = [r for r in grp if r["nb"] is not None and r["nb"]["side"]]
    print(f"\n{name}: {len(has)}/{len(grp)} have a side-by-side parallel neighbour")
