#!/usr/bin/env python
"""How well can superpoints represent a ground-truth part AT ALL?

Snapping masks to the superpoints from make_superpoints.py collapsed AP50 from
0.5716 to 0.19-0.24 at every vote threshold. Before tuning the threshold, ask
the question that should have come first: if we could choose the BEST possible
set of superpoints for each ground-truth instance, how close could we get?

That oracle is the ceiling of the representation. A superpoint straddling a part
boundary makes the part unreachable no matter how the vote is tuned, and no
downstream cleverness recovers it.

For each GT instance, take every superpoint whose overlap with it exceeds
`vote`, union them, and measure IoU against the instance. Report the
distribution. If the median oracle IoU sits below what our masks already
achieve, the granularity is simply too coarse and the idea is dead as
implemented.
"""
import argparse
import os
from pathlib import Path

import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp", default="/data/challenge/analysis/superpoints/validation")
    ap.add_argument("--vote", type=float, default=0.5)
    args = ap.parse_args()

    spd = Path(args.sp)
    ious, sizes = [], []
    straddle = 0
    for f in sorted(spd.glob("*.npy")):
        scene = f.stem
        npy = os.path.join(VAL, scene + ".npy")
        if not os.path.exists(npy):
            continue
        a = np.load(npy, mmap_mode="r")
        inst = np.asarray(a[:, 10]).astype(int)
        seg = np.load(f)
        nseg = int(seg.max()) + 1
        seg_size = np.bincount(seg, minlength=nseg).astype(np.float64)
        for gi in np.unique(inst):
            if gi < 0:
                continue
            gm = inst == gi
            g = int(gm.sum())
            if g < 10:
                continue
            hit = np.bincount(seg[gm], minlength=nseg).astype(np.float64)
            frac = hit / np.maximum(seg_size, 1)
            keep = frac > args.vote
            pred = keep[seg]
            inter = int((pred & gm).sum())
            union = int((pred | gm).sum())
            ious.append(inter / max(union, 1))
            sizes.append(g)
            # a superpoint is "straddling" when it is substantially split by the
            # instance boundary -- the failure mode that caps this approach
            straddle += int(((frac > 0.2) & (frac < 0.8)).sum())

    v = np.array(ious)
    print(f"GT instances: {len(v)}   vote={args.vote}\n")
    print(f"oracle IoU  median {np.median(v):.3f}   mean {v.mean():.3f}")
    for t in (0.5, 0.7, 0.9, 0.95):
        print(f"  >= {t:.2f}: {100*np.mean(v >= t):5.1f}%")
    print(f"\nstraddling superpoints (20-80% inside an instance): {straddle}")
    small = np.array(sizes) < 500
    if small.any():
        print(f"oracle IoU on small parts (<500 pts, n={int(small.sum())}): "
              f"median {np.median(v[small]):.3f}")
        print(f"           on large parts (n={int((~small).sum())}): "
              f"median {np.median(v[~small]):.3f}")
    print("\nOur predicted masks already reach IoU>=0.5 on 63.8% of instances.")
    print("If the oracle above is not far higher than that, superpoints at this")
    print("granularity cannot help and the granularity must change.")


if __name__ == "__main__":
    main()
