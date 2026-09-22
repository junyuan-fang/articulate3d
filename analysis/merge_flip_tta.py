#!/usr/bin/env python
"""Merge original and x-mirrored inference into one prediction set (flip-TTA).

Rows are aligned by construction (mirroring does not reorder points), so masks
from the flipped run apply to the original cloud as-is. Geometry is mirrored
back with the per-scene constant C recorded by make_flipped_val.py:
    origin_x -> C - origin_x        axis_x -> -axis_x

Merging simply concatenates the two query sets (up to 200 instances/scene --
the submission format allows any N). Optional IoU-based NMS drops the
lower-scoring near-duplicates; AP ranks by score, so duplicates mostly waste
precision at the tail, but we measure both.

Usage:
  merge_flip_tta.py ORIG.pkl FLIP.pkl OUT.pkl --consts flip_constants.json [--nms 0.8]
"""
import argparse
import json
import pickle

import numpy as np


def nms_masks(masks, scores, keep_idx, thresh):
    """Greedy IoU-NMS over boolean mask columns. Returns surviving indices."""
    order = np.argsort(-scores)
    alive = []
    for i in order:
        mi = masks[:, i]
        si = mi.sum()
        if si == 0:
            continue
        dup = False
        for j in alive:
            inter = (mi & masks[:, j]).sum()
            union = si + masks[:, j].sum() - inter
            if union and inter / union > thresh:
                dup = True
                break
        if not dup:
            alive.append(i)
    return np.array(sorted(alive), dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig")
    ap.add_argument("flip")
    ap.add_argument("out")
    ap.add_argument("--consts", required=True)
    ap.add_argument("--nms", type=float, default=0.0, help="IoU threshold; 0 = keep all 200")
    ap.add_argument("--addnew", type=float, default=0.0,
                    help="selective union: keep all originals untouched and append only "
                         "flipped queries whose IoU with EVERY original mask is below this "
                         "threshold, at score x 0.5 (mirrored run is weaker: recall 55.9 "
                         "vs 70.5; blind concat floods the ranking with weak duplicates)")
    args = ap.parse_args()

    A = pickle.load(open(args.orig, "rb"))
    B = pickle.load(open(args.flip, "rb"))
    C = json.load(open(args.consts))
    assert set(A) == set(B), "scene sets differ"

    out = {}
    tot_kept = tot_all = 0
    # pop as we go: holding both inputs AND the merged output peaks over 20 GB
    # on this 31 GB box (the first attempt was OOM-killed); releasing consumed
    # scenes keeps the peak near a single copy of the data.
    for scene in sorted(A.keys()):
        a, b = A.pop(scene), B.pop(scene)
        c = C[scene]
        # mirror flip-run geometry back into the original frame
        bo = np.asarray(b["pred_origins"], dtype=np.float64).copy()
        bo[:, 0] = c - bo[:, 0]
        ba = np.asarray(b["pred_axises"], dtype=np.float64).copy()
        ba[:, 0] = -ba[:, 0]

        am = np.asarray(a["pred_masks"])
        bm = np.asarray(b["pred_masks"])
        bs = np.asarray(b["pred_scores"]).ravel()
        if args.addnew > 0:
            abin = am > 1e-3
            bbin = bm > 1e-3
            asum = abin.sum(0)
            keep_b = []
            for qi in range(bbin.shape[1]):
                m = bbin[:, qi]
                s = m.sum()
                if s == 0:
                    continue
                inter = abin[m].sum(0)                       # vs every original
                union = asum + s - inter
                iou = np.where(union > 0, inter / union, 0)
                if iou.max() < args.addnew:
                    keep_b.append(qi)
            keep_b = np.array(keep_b, dtype=int)
            bm = bm[:, keep_b]
            bs = bs[keep_b] * 0.5                            # rank below originals
            bo, ba = bo[keep_b], ba[keep_b]
            bcls = np.asarray(b["pred_classes"]).ravel()[keep_b]
        else:
            bcls = np.asarray(b["pred_classes"]).ravel()

        masks = np.concatenate([am, bm], axis=1)
        scores = np.concatenate([np.asarray(a["pred_scores"]).ravel(), bs])
        classes = np.concatenate([np.asarray(a["pred_classes"]).ravel(), bcls])
        origins = np.concatenate([np.asarray(a["pred_origins"], dtype=np.float64), bo])
        axes = np.concatenate([np.asarray(a["pred_axises"], dtype=np.float64), ba])

        tot_all += masks.shape[1]
        if args.nms > 0:
            keep = nms_masks(masks > 1e-3, scores, None, args.nms)
            masks, scores, classes = masks[:, keep], scores[keep], classes[keep]
            origins, axes = origins[keep], axes[keep]
        tot_kept += masks.shape[1]

        out[scene] = {"pred_masks": masks.astype(np.float32),
                      "pred_scores": scores.astype(np.float32),
                      "pred_classes": classes.astype(np.int64),
                      "pred_origins": origins.astype(np.float32),
                      "pred_axises": axes.astype(np.float32)}

    pickle.dump(out, open(args.out, "wb"))
    print(f"{len(out)} scenes | instances {tot_all} -> {tot_kept}"
          f"{f' (NMS@{args.nms})' if args.nms > 0 else ' (no NMS)'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
