#!/usr/bin/env python
"""Offline DBSCAN-split screen for MOV masks (mask-only M_ap50, official
evaluator). Masks changing means the geometry stack must be re-run before any
submission — this script only screens whether splitting helps mask AP50.

Mov parts are big (doors/drawers); merged double-doors may split at the frame
gap if eps is small enough, but a mask that bridges the gap via the frame
won't. Screen eps in {0.05, 0.08, 0.12} + fragment floor.
"""
import argparse, os, pickle, sys
import numpy as np

sys.path.insert(0, "/data/challenge/USDNet")
from benchmark.evaluate_semantic_instance import evaluate
from sklearn.cluster import DBSCAN

DATA = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
GT = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/instance_gt/validation"
OUT = "/data/challenge/scratch/mov_pp_tmp"
MASK_TH = 0.5

# keep pred_origins/pred_axises columns aligned when masks are dropped/split


def xyz_of(scene):
    a = np.load(os.path.join(DATA, scene + ".npy"), mmap_mode="r")
    return np.asarray(a[:, :3])


def apply(d, scene, eps, min_size, eps_by_class=None):
    # eps_by_class: {class_id: eps} overrides the shared eps per prediction
    m = d["pred_masks"]; s = d["pred_scores"]; c = d["pred_classes"]
    org = d.get("pred_origins"); axi = d.get("pred_axises")
    mb = m > MASK_TH
    xyz = xyz_of(scene)
    cols, scs, cls, orgs, axis_ = [], [], [], [], []
    for i in range(m.shape[1]):
        eps_i = eps if eps_by_class is None else eps_by_class.get(int(c[i]), eps)
        mi = mb[:, i]
        n = int(mi.sum())
        if n < min_size:
            continue
        sub = xyz[mi]
        if n > 60000:
            comps = [np.arange(n)]
        else:
            lab = DBSCAN(eps=eps_i, min_samples=1, n_jobs=4).fit(sub).labels_
            comps = [np.where(lab == u)[0] for u in np.unique(lab)]
        idx = np.where(mi)[0]
        for comp in comps:
            if len(comp) < min_size:
                continue
            nm = np.zeros(m.shape[0], np.float32)
            nm[idx[comp]] = 1.0
            cols.append(nm); scs.append(s[i]); cls.append(c[i])
            if org is not None: orgs.append(org[i])
            if axi is not None: axis_.append(axi[i])
    out = {"pred_masks": np.stack(cols, 1), "pred_scores": np.array(scs, np.float32),
           "pred_classes": np.array(cls, np.int64)}
    if org is not None: out["pred_origins"] = np.stack(orgs)
    if axi is not None: out["pred_axises"] = np.stack(axis_)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="/data/challenge/USDNet/results/infer_val_gold/debug/val_preds/preds.pkl")
    ap.add_argument("--variants", nargs="*", default=["base", "0.05:50", "0.08:50", "0.12:50"])
    args = ap.parse_args()
    preds = pickle.load(open(args.pkl, "rb"))
    os.makedirs(OUT, exist_ok=True)
    for name in args.variants:
        if name == "base":
            v = preds
        else:
            eps, ms = name.split(":")
            v = {sc: apply(d, sc, float(eps), int(ms)) for sc, d in preds.items()}
        ap50 = evaluate(v, GT, os.path.join(OUT, f"pred_{name.replace(':','_')}"),
                        dataset="articulate3d", eval_articulation=False,
                        gt_articulations={}, eval_hierarchy_inter=False)
        print(f"VARIANT {name}: M_ap50 = {ap50}", flush=True)
