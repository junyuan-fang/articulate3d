#!/usr/bin/env python
"""Two candidate gains for the final submission.

1. Class-confusion audit: among IoU>0.5 matches, how often is the predicted
   rotation/translation class wrong? Each error is a double hit (FP in the
   predicted class + unmatched GT in the true class).

2. Geometric-confidence reranking: raw scores barely correlate with
   articulation correctness (r=+0.036 measured earlier), yet AP integrates
   the ranked list. Two deployable signals, no GT involved:
     - rotations: how far the raw predicted origin had to move to reach its
       snapped endpoint (short move = the model already agreed = confident);
     - translations: the angle between the raw axis and the PCA panel normal
       (small = two independent estimators agree).
   new_score = score * (0.5 + 0.5 * exp(-signal / tau)).
"""
import pickle

import numpy as np

RAW = "/data/challenge/USDNet/results/ev_v4init/debug/val_preds/preds.pkl"
FINAL = "/tmp/v4init_full.pkl"          # 0.4306 stack (pca+endpoint+handles)
VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"

import h5py, os

raw = pickle.load(open(RAW, "rb"))
fin = pickle.load(open(FINAL, "rb"))

# ---------- 1. class confusion ----------
cm = np.zeros((2, 2), dtype=int)        # [gt-1][pred-1]
for scene in sorted(raw):
    inst = np.asarray(np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")[:, 10]).astype(int)
    gt = {}
    with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
        for k in h.keys():
            gt[int(k)] = int(np.asarray(h[k]["sem_id"]))
    if not gt:
        continue
    gmasks = {i: inst == i for i in gt}
    pm = np.asarray(raw[scene]["pred_masks"]) > 1e-3
    cls = np.asarray(raw[scene]["pred_classes"]).ravel()
    order = np.argsort(-np.asarray(raw[scene]["pred_scores"]).ravel())
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
            cm[gt[bid] - 1][cls[qi] - 1] += 1

n = cm.sum()
print("=== class confusion on matched instances ===")
print(f"  GT rot  -> pred rot {cm[0,0]:3d}   pred tr {cm[0,1]:3d}")
print(f"  GT tr   -> pred rot {cm[1,0]:3d}   pred tr {cm[1,1]:3d}")
print(f"  wrong-class rate: {(cm[0,1]+cm[1,0])/max(n,1)*100:.1f}%  ({cm[0,1]+cm[1,0]}/{n})")

# ---------- 2. rerank ----------
import copy


def rerank(fin_preds, raw_preds, tau_d=0.3, tau_a=20.0):
    out = {}
    for scene in fin_preds:
        p = copy.deepcopy(fin_preds[scene])
        r = raw_preds[scene]
        cls = np.asarray(p["pred_classes"]).ravel()
        s = np.asarray(p["pred_scores"], dtype=np.float64).copy()
        ro = np.asarray(r["pred_origins"], dtype=np.float64)
        fo = np.asarray(p["pred_origins"], dtype=np.float64)
        ra = np.asarray(r["pred_axises"], dtype=np.float64)
        fa = np.asarray(p["pred_axises"], dtype=np.float64)
        d = np.linalg.norm(fo - ro, axis=1)                       # origin move
        rn = ra / np.maximum(np.linalg.norm(ra, axis=1, keepdims=True), 1e-9)
        fn = fa / np.maximum(np.linalg.norm(fa, axis=1, keepdims=True), 1e-9)
        ang = np.degrees(np.arccos(np.clip(np.abs((rn * fn).sum(1)), 0, 1)))  # axis change
        conf = np.ones_like(s)
        rot = cls == 1
        conf[rot] = np.exp(-d[rot] / tau_d)
        tr = cls == 2
        conf[tr] = np.exp(-ang[tr] / tau_a)
        p["pred_scores"] = (s * (0.5 + 0.5 * conf)).astype(np.float32)
        out[scene] = p
    return out


import sys
sys.path.insert(0, "/data/challenge/USDNet")
sys.path.insert(0, "/data/challenge/analysis")
import build_matches as bm
import benchmark.evaluate_semantic_instance as esi


def score(preds, tag):
    tmp = f"/tmp/_rr_{tag}.pkl"
    pickle.dump(preds, open(tmp, "wb"))
    bm.PRED_PKL = tmp
    matches, _, _ = bm.build()
    vals = []
    for crit in (esi.match_criteria_M_pred, esi.match_criteria_MA_pred,
                 esi.match_criteria_MO_pred, esi.match_criteria_MAO_pred_standard):
        ap = esi.compute_averages(esi.evaluate_matches(matches, crit))
        vals.append(ap["all_ap_50%"])
    print(f"  {tag:14s} AP50={vals[0]:.4f} axis={vals[1]:.4f} origin={vals[2]:.4f} BOTH={vals[3]:.4f}")
    return vals[3]


print("\n=== rerank sweep (baseline stack BOTH=0.4306) ===")
best = (None, 0.4306)
for tau_d, tau_a in [(0.2, 15.0), (0.3, 20.0), (0.5, 30.0)]:
    b = score(rerank(fin, raw, tau_d, tau_a), f"td{tau_d}_ta{tau_a}")
    if b > best[1]:
        best = ((tau_d, tau_a), b)
print("best:", best)
