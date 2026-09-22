#!/usr/bin/env python
"""Second sweep: how much is left on the table once the origin is snapped to a
mask endpoint perpendicular to the axis?

`perp` picks between the two endpoints by "whichever the raw predicted origin is
closer to". The oracle variants replace that choice with the GT-optimal one and
bound what a *learned* selector could buy. Oracle rows use GT and exist only to
size the prize -- they are never submitted.

Usage: python pp_sweep2.py [preds.pkl]
"""
import os
import sys
import pickle
from pathlib import Path

import numpy as np

import build_matches as bm
import benchmark.evaluate_semantic_instance as esi
from pp_sweep import (POINTS_DIR, Z, ROTATION, MASK_THRESH, normalize,
                      snap_canonical, build_matches_from, score)


def perp_endpoints(pts, axis, qlo=5.0, qhi=95.0):
    """Two robust endpoints of the mask along its widest direction PERPENDICULAR
    to `axis` -- i.e. the two lateral edges of a door hinged about `axis`."""
    a = normalize(np.asarray(axis, np.float64))
    c = pts.mean(0)
    X = pts - c
    Xp = X - np.outer(X @ a, a)
    _, _, vt = np.linalg.svd(Xp, full_matrices=False)
    d = normalize(vt[0] - (vt[0] @ a) * a)
    t = Xp @ d
    lo, hi = np.percentile(t, [qlo, qhi])
    return np.stack([c + lo * d, c + hi * d]), d


def line_dist(p, o, a):
    """Perpendicular distance from point p to the line (o, a) -- the metric."""
    a = normalize(np.asarray(a, np.float64))
    return np.linalg.norm(np.cross(p - o, a))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else (
        "/data/challenge/USDNet/results/eval_mov_val_einsum/debug/val_preds/preds.pkl")
    raw = pickle.load(open(src, "rb"))
    pts_cache = {s: np.load(POINTS_DIR / f"{s}.npy", mmap_mode="r")[:, 0:3]
                 .astype(np.float64) for s in raw}
    print(f"loaded {len(raw)} scenes", flush=True)

    # ---- axis pre-processing common to every row below -------------------
    base = {}
    for s, p in raw.items():
        ax = snap_canonical(normalize(np.asarray(p["pred_axises"], np.float64)),
                            p["pred_classes"])
        base[s] = dict(p, pred_axises=ax)

    # ---- IoU>0.5 matching, done directly on query index so the oracles can
    # look up the GT belonging to each *query slot* of preds.pkl ------------
    bm.set_articulate3d_labels()
    gt_for = {}          # (scene, query_idx) -> (gt_origin, gt_axis)
    for s, p in base.items():
        gt_ids = np.loadtxt(os.path.join(bm.GT_TXT_DIR, s + ".txt"), dtype=np.int64)
        arti = bm.load_gt_articulation(s)
        masks = np.asarray(p["pred_masks"]) > MASK_THRESH
        for gid, a in arti.items():
            g = gt_ids == gid
            n_g = g.sum()
            if n_g == 0:
                continue
            inter = (masks & g[:, None]).sum(0)
            iou = inter / (n_g + masks.sum(0) - inter + 1e-9)
            q = int(np.argmax(iou))
            if iou[q] > 0.5:
                gt_for[(s, q)] = (a["origin"], a["axis"])
    print(f"oracle matches: {len(gt_for)}", flush=True)

    def variant(mode, qlo=5.0, qhi=95.0):
        out = {}
        stats = []
        for s, p in base.items():
            pts = pts_cache[s]
            org = np.asarray(p["pred_origins"], np.float64).copy()
            cls = np.asarray(p["pred_classes"]).ravel()
            masks = np.asarray(p["pred_masks"])
            for i in range(org.shape[0]):
                if cls[i] != ROTATION:
                    continue
                sel = masks[:, i] > MASK_THRESH
                if sel.sum() < 3:
                    continue
                mp = pts[sel]
                cand, _ = perp_endpoints(mp, p["pred_axises"][i], qlo, qhi)
                if mode == "perp":
                    org[i] = cand[int(np.argmin(np.linalg.norm(cand - org[i], axis=1)))]
                elif mode == "perp_oracle":
                    g = gt_for.get((s, i))
                    if g is None:
                        org[i] = cand[int(np.argmin(np.linalg.norm(cand - org[i], axis=1)))]
                    else:
                        d = [line_dist(c, g[0], g[1]) for c in cand]
                        org[i] = cand[int(np.argmin(d))]
                        stats.append(min(d))
                elif mode == "anypoint_oracle":
                    g = gt_for.get((s, i))
                    if g is None:
                        continue
                    a = normalize(np.asarray(g[1], np.float64))
                    d = np.linalg.norm(np.cross(mp - g[0], a), axis=1)
                    org[i] = mp[int(np.argmin(d))]
                    stats.append(d.min())
                elif mode == "perp+near":
                    e = cand[int(np.argmin(np.linalg.norm(cand - org[i], axis=1)))]
                    org[i] = mp[int(np.argmin(np.linalg.norm(mp - e, axis=1)))]
            out[s] = dict(p, pred_origins=org)
        if stats:
            st = np.asarray(stats)
            print(f"    [{mode}] n={len(st)} median={np.median(st):.3f} "
                  f"frac<0.25={np.mean(st < 0.25):.3f}", flush=True)
        return out

    score(base, "norm+canon (no origin step)")
    score(variant("perp"), "+ endpt(perp,5/95)  [pred choice]")
    score(variant("perp+near"), "+ endpt(perp)->nearest real point")
    score(variant("perp_oracle"), "+ endpt(perp)  [ORACLE choice]")
    score(variant("anypoint_oracle"), "+ best mask point [ORACLE]")


if __name__ == "__main__":
    main()
