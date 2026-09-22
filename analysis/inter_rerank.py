#!/usr/bin/env python
"""Interactable-track prior: a handle sits ON a movable part.

The inter phase scores plain AP50 over interactable-part masks, and AP is
rank-sensitive -- so demoting predictions that lie outside every predicted
movable part should lift it without touching any mask. Movable masks come from
our own mov model on the same scenes, so the rule is deployable at test time.

Signal per inter query: fraction of its points that fall inside the union of
movable-part masks (score > MOV_SCORE). new_score = score * (w + (1-w)*frac).

Usage: inter_rerank.py INTER.pkl MOV.pkl OUT.pkl [--w 0.3] [--mov-score 0.5]
"""
import argparse
import pickle

import numpy as np

MASK_TH = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inter")
    ap.add_argument("mov")
    ap.add_argument("out")
    ap.add_argument("--w", type=float, default=0.3,
                    help="floor multiplier for zero-overlap predictions")
    ap.add_argument("--mov-score", type=float, default=0.5)
    args = ap.parse_args()

    I = pickle.load(open(args.inter, "rb"))
    Mv = pickle.load(open(args.mov, "rb"))
    stats = []
    out = {}
    for scene in sorted(I):
        p = I[scene]
        im = np.asarray(p["pred_masks"]) > MASK_TH
        isc = np.asarray(p["pred_scores"], dtype=np.float64).ravel()
        if scene not in Mv:
            out[scene] = p
            continue
        mv = Mv[scene]
        mm = np.asarray(mv["pred_masks"]) > MASK_TH
        msc = np.asarray(mv["pred_scores"]).ravel()
        keep = msc > args.mov_score
        union = mm[:, keep].any(axis=1) if keep.any() else np.zeros(mm.shape[0], bool)

        n = im.sum(0)
        inside = (im & union[:, None]).sum(0)
        frac = np.where(n > 0, inside / np.maximum(n, 1), 0.0)
        stats.append(frac)
        p = dict(p)
        p["pred_scores"] = (isc * (args.w + (1 - args.w) * frac)).astype(np.float32)
        out[scene] = p

    pickle.dump(out, open(args.out, "wb"))
    f = np.concatenate(stats) if stats else np.array([0.0])
    print(f"{len(out)} scenes | inside-movable fraction: "
          f"median {np.median(f):.3f}  >0.5: {np.mean(f>0.5)*100:.1f}%  ==0: {np.mean(f==0)*100:.1f}%")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
