#!/usr/bin/env python
"""Memory-lean rewrite of rerank_and_confusion.py (v1 was OOM-killed at ~20 GB
alongside training: raw + stack + deepcopy + the scorer's own reload).

Pass 1 extracts tiny per-query signal arrays and the confusion matrix, then
frees everything big. Each rerank variant then re-reads the stack pkl from
disk, patches scores in place, dumps, frees, and lets the scorer load from
disk -- one full copy in memory at any time.
"""
import os
import pickle
import sys

import h5py
import numpy as np

RAW = "/data/challenge/USDNet/results/ev_v4init/debug/val_preds/preds.pkl"
FINAL = "/tmp/v4init_full.pkl"
VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"

sys.path.insert(0, "/data/challenge/USDNet")
sys.path.insert(0, "/data/challenge/analysis")
import build_matches as bm
import benchmark.evaluate_semantic_instance as esi

# ---------- pass 1: signals + confusion ----------
signals = {}
cm = np.zeros((2, 2), dtype=int)
raw = pickle.load(open(RAW, "rb"))
fin = pickle.load(open(FINAL, "rb"))
for scene in sorted(raw):
    r, p = raw[scene], fin[scene]
    cls = np.asarray(p["pred_classes"]).ravel()
    ro = np.asarray(r["pred_origins"], dtype=np.float64)
    fo = np.asarray(p["pred_origins"], dtype=np.float64)
    ra = np.asarray(r["pred_axises"], dtype=np.float64)
    fa = np.asarray(p["pred_axises"], dtype=np.float64)
    d = np.linalg.norm(fo - ro, axis=1)
    rn = ra / np.maximum(np.linalg.norm(ra, axis=1, keepdims=True), 1e-9)
    fn = fa / np.maximum(np.linalg.norm(fa, axis=1, keepdims=True), 1e-9)
    ang = np.degrees(np.arccos(np.clip(np.abs((rn * fn).sum(1)), 0, 1)))
    signals[scene] = (cls.copy(), d, ang,
                     np.asarray(p["pred_scores"], dtype=np.float64).copy())

    inst = np.asarray(np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")[:, 10]).astype(int)
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            gt[int(k)] = int(np.asarray(h[k]["sem_id"]))
    if gt:
        gmasks = {i: inst == i for i in gt}
        pm = np.asarray(r["pred_masks"]) > 1e-3
        order = np.argsort(-np.asarray(r["pred_scores"]).ravel())
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
            if best > 0.5 and bid is not None:
                used.add(bid)
                cm[gt[bid] - 1][int(np.asarray(r["pred_classes"]).ravel()[qi]) - 1] += 1
del raw, fin

n = cm.sum()
print("=== class confusion on matched instances ===")
print(f"  GT rot -> pred rot {cm[0,0]:3d} | pred tr {cm[0,1]:3d}")
print(f"  GT tr  -> pred rot {cm[1,0]:3d} | pred tr {cm[1,1]:3d}")
print(f"  wrong-class: {(cm[0,1]+cm[1,0])}/{n} = {(cm[0,1]+cm[1,0])/max(n,1)*100:.1f}%")


def score_variant(tau_d, tau_a, tag):
    fin2 = pickle.load(open(FINAL, "rb"))
    for scene, p in fin2.items():
        cls, d, ang, s = signals[scene]
        conf = np.ones_like(s)
        conf[cls == 1] = np.exp(-d[cls == 1] / tau_d)
        conf[cls == 2] = np.exp(-ang[cls == 2] / tau_a)
        p["pred_scores"] = (s * (0.5 + 0.5 * conf)).astype(np.float32)
    tmp = f"/tmp/_rr_{tag}.pkl"
    pickle.dump(fin2, open(tmp, "wb"))
    del fin2
    bm.PRED_PKL = tmp
    matches, _, _ = bm.build()
    ap = esi.compute_averages(esi.evaluate_matches(matches, esi.match_criteria_MAO_pred_standard))
    both = ap["all_ap_50%"]
    ap50 = esi.compute_averages(esi.evaluate_matches(matches, esi.match_criteria_M_pred))["all_ap_50%"]
    print(f"  tau_d={tau_d} tau_a={tau_a}:  AP50={ap50:.4f}  BOTH={both:.4f}")
    os.remove(tmp)
    return both


print("\n=== rerank sweep (baseline stack BOTH=0.4306) ===")
for td, ta in [(0.2, 15.0), (0.3, 20.0), (0.5, 30.0)]:
    score_variant(td, ta, f"{td}_{ta}")
