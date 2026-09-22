#!/usr/bin/env python
"""Can a RELATION detect bottom-hinged doors, where part shape cannot?

diag_lateral_geom.py confirms no single-part feature separates lateral from
vertical hinges -- attitude, planarity, height and width all overlap. That is
the same wall the sliding-door problem hit, and there the answer was to stop
asking about the part and start asking about its neighbours.

Two relational hypotheses, both with a physical story:

  H1 stacking. A bottom-hinged door (oven, dishwasher, drop-down cabinet)
     belongs to a VERTICAL column of coplanar fronts; a side-hinged door
     belongs to a HORIZONTAL row of them. So compare, over coplanar
     neighbours, whether separation is mostly vertical or mostly lateral.

  H2 obstruction above. A bottom-hinged door has structure directly above it
     -- a countertop, the next appliance -- because that is why it cannot swing
     sideways. A full-height side-hinged door has open space above.

Both are measured against the same 7 lateral / 173 vertical ground truth.
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


def flat_normal(pts):
    c = pts - pts.mean(0)
    s = c if len(c) <= 4000 else c[np.random.RandomState(0).choice(len(c), 4000, False)]
    _, _, vt = np.linalg.svd(s, full_matrices=False)
    n = vt[2].copy()
    n[2] = 0.0
    nn = np.linalg.norm(n)
    return (n / nn) if nn > 1e-6 else None


preds = pickle.load(open(PKL, "rb"))
rows = []
for scene in sorted(preds):
    a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
    pts_all = np.asarray(a[:, 0:3])
    inst = np.asarray(a[:, 10]).astype(int)
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            g = h[k]
            if int(np.asarray(g["sem_id"])) == ROT:
                gt[int(k)] = np.asarray(g["axis"], float)
    if not gt:
        continue
    gmasks = {i: inst == i for i in gt}
    pm = np.asarray(preds[scene]["pred_masks"]) > MASK_TH
    sc = np.asarray(preds[scene]["pred_scores"]).ravel()

    # all reasonably-confident predicted parts in the scene, for neighbour search
    cand = []
    for qi in np.where(sc > 0.3)[0]:
        sel = pm[:, qi]
        if 50 <= sel.sum():
            p = pts_all[sel]
            cand.append((p.mean(0), flat_normal(p), p))

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
        pts = pts_all[m]
        if len(pts) < 10:
            continue
        nrm = flat_normal(pts)
        if nrm is None:
            continue
        cen = pts.mean(0)
        inp = np.array([-nrm[1], nrm[0], 0.0])

        # H1: among coplanar neighbours, is the arrangement vertical or lateral?
        vstack = lstack = 0
        for ocen, onrm, _ in cand:
            if onrm is None or abs(float(np.dot(nrm, onrm))) < 0.94:
                continue
            d = ocen - cen
            if abs(float(d @ nrm)) > 0.15:          # not the same facade
                continue
            dz, di = abs(float(d[2])), abs(float(d @ inp))
            if dz < 0.05 and di < 0.05:
                continue                             # itself
            if dz > di:
                vstack += 1
            else:
                lstack += 1

        # H2: is there scene structure directly above the part's top edge?
        top = pts[:, 2].max()
        band = pts_all[(pts_all[:, 2] > top + 0.03) & (pts_all[:, 2] < top + 0.30)]
        above = 0
        if len(band):
            lo, hi = pts[:, 0:2].min(0), pts[:, 0:2].max(0)
            inside = np.all((band[:, 0:2] > lo) & (band[:, 0:2] < hi), axis=1)
            above = int(inside.sum())
        occ = above / max(len(pts), 1)

        gax = gt[bid]
        rows.append(dict(scene=scene, lateral=abs(float(gax[2] / max(np.linalg.norm(gax), 1e-9))) < 0.5,
                         vstack=vstack, lstack=lstack, occ=occ, score=float(sc[qi])))

lat = [r for r in rows if r["lateral"]]
ver = [r for r in rows if not r["lateral"]]
print(f"matched rotations {len(rows)}:  lateral {len(lat)}   vertical {len(ver)}\n")

print("H1 stacking  (vstack = coplanar neighbours above/below, lstack = beside)")
for name, g in (("lateral ", lat), ("vertical", ver)):
    v = np.array([r["vstack"] for r in g]); l = np.array([r["lstack"] for r in g])
    print(f"  {name}  vstack med {np.median(v):4.1f}  lstack med {np.median(l):4.1f}  "
          f"vstack>lstack in {int((v > l).sum())}/{len(g)}")

print("\nH2 obstruction above  (points in a 3-30 cm band over the part / part size)")
for name, g in (("lateral ", lat), ("vertical", ver)):
    o = np.array([r["occ"] for r in g])
    print(f"  {name}  median {np.median(o):6.3f}   [{o.min():.3f} {o.max():.3f}]")

print("\nper-part detail (lateral):")
for r in sorted(lat, key=lambda x: -x["occ"]):
    print(f"  {r['scene']:12s} vstack {r['vstack']:2d}  lstack {r['lstack']:2d}  "
          f"occ {r['occ']:6.3f}")
