#!/usr/bin/env python
"""Compute a geometric over-segmentation (superpoints) for each scene.

gate_superpoint_headroom.py found 24 validation instances whose best predicted
mask lands in IoU [0.40, 0.50) -- near misses that a boundary cleanup could push
over the AP50 threshold. We need only +0.8% relative AP50 to take the last
leaderboard sub-metric, so this is the cheapest remaining lever.

The preprocessed .npy has a segments column (index 11) but it is ALL ZEROS --
superpoints were never computed, so the repo's `eval_on_segments` switch would
treat each scene as a single segment. Rather than rely on the mesh files (whose
vertex order need not match the preprocessed points), we segment the point cloud
directly: it already carries normals in columns 6-8.

Felzenszwalb graph segmentation over a kNN graph, edge weight = normal
dissimilarity -- the same family as the ScanNet over-segmentation Mask3D was
designed around. Segments are then merged up to a minimum size so a mask vote
inside one is meaningful.

Writes {scene}.npy of int32 segment ids, one per point, to --out.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)
        self.internal = np.zeros(n, dtype=np.float32)

    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b, w):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.internal[ra] = w
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def segment_scene(xyz, normals, k=10, thresh=0.06, min_size=40):
    n = len(xyz)
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1, workers=-1)
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].ravel()
    # undirected, deduplicated
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    edges = np.unique(np.stack([lo, hi], 1), axis=0)

    nrm = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    w = 1.0 - np.abs((nrm[edges[:, 0]] * nrm[edges[:, 1]]).sum(1))
    # break ties toward spatial proximity so flat regions do not merge across gaps
    w = w + 0.5 * np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)

    order = np.argsort(w)
    edges, w = edges[order], w[order]

    uf = UnionFind(n)
    for (a, b), wt in zip(edges, w):
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        # Felzenszwalb predicate: merge when the edge is cheaper than both
        # components' internal variation plus a size-dependent slack
        ta = uf.internal[ra] + thresh / uf.size[ra]
        tb = uf.internal[rb] + thresh / uf.size[rb]
        if wt <= min(ta, tb):
            uf.union(ra, rb, wt)

    # absorb tiny segments into their cheapest neighbour
    for (a, b), wt in zip(edges, w):
        ra, rb = uf.find(a), uf.find(b)
        if ra != rb and (uf.size[ra] < min_size or uf.size[rb] < min_size):
            uf.union(ra, rb, wt)

    roots = np.array([uf.find(i) for i in range(n)])
    _, seg = np.unique(roots, return_inverse=True)
    return seg.astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--data", default="/data/challenge/USDNet/data/processed/articulate3d_challenge_mov")
    ap.add_argument("--out", default="/data/challenge/analysis/superpoints")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--thresh", type=float, default=0.06)
    ap.add_argument("--min-size", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.data) / args.split
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    scenes = sorted(p.stem for p in src.glob("*.npy"))
    if args.limit:
        scenes = scenes[:args.limit]

    for i, s in enumerate(scenes):
        dst = out / f"{s}.npy"
        if dst.exists():
            continue
        t0 = time.time()
        a = np.load(src / f"{s}.npy", mmap_mode="r")
        seg = segment_scene(np.asarray(a[:, 0:3], dtype=np.float64),
                            np.asarray(a[:, 6:9], dtype=np.float64),
                            k=args.k, thresh=args.thresh, min_size=args.min_size)
        np.save(dst, seg)
        sizes = np.bincount(seg)
        print(f"[{i+1}/{len(scenes)}] {s}  pts {len(seg):7d}  segments {seg.max()+1:6d}  "
              f"median size {int(np.median(sizes)):5d}  {time.time()-t0:5.1f}s", flush=True)


if __name__ == "__main__":
    main()
