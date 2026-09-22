#!/usr/bin/env python
"""Offline postprocess grid for the inter task, scored with the OFFICIAL
benchmark evaluator (same code path the trainer/server pipeline uses).

Variants are built lazily per scene to keep one 5GB preds copy in RAM.
DBSCAN splitting runs on CPU from the dumped masks + xyz (no GPU needed).

Usage: python inter_pp_grid.py [--pkl PATH] [--variant NAME] (default: run all)
"""
import argparse, os, pickle, sys
import numpy as np

sys.path.insert(0, "/data/challenge/USDNet")
from benchmark.evaluate_semantic_instance import evaluate
from sklearn.cluster import DBSCAN

DATA = "/data/challenge/USDNet/data/processed/articulate3d_challenge_inter/validation"
GT = "/data/challenge/USDNet/data/processed/articulate3d_challenge_inter/instance_gt/validation"
OUT = "/data/challenge/scratch/inter_pp_tmp"
MASK_TH = 0.5  # masks are stored as 0/1 float32


def xyz_of(scene):
    a = np.load(os.path.join(DATA, scene + ".npy"), mmap_mode="r")
    return np.asarray(a[:, :3])


def variant_scene(d, scene, name):
    m = d["pred_masks"]; s = d["pred_scores"].copy(); c = d["pred_classes"].copy()
    mb = m > MASK_TH
    sizes = mb.sum(0)
    if name == "base":
        return d
    if name.startswith("minsize"):
        n = int(name[len("minsize"):])
        keep = sizes >= n
        return {"pred_masks": m[:, keep], "pred_scores": s[keep], "pred_classes": c[keep]}
    if name.startswith("dil"):
        # dilate masks: dil<r> (all classes) or dil2_<r> (class 2 only)
        from scipy.spatial import cKDTree
        if name.startswith("dil2_"):
            r = float(name[5:]); only2 = True
        else:
            r = float(name[3:]); only2 = False
        xyz = xyz_of(scene)
        tree = cKDTree(xyz)
        cols = []
        for i in range(m.shape[1]):
            mi = mb[:, i]
            if mi.sum() == 0 or mi.sum() > 5000 or (only2 and int(c[i]) != 2):
                cols.append(m[:, i]); continue
            idx = tree.query_ball_point(xyz[mi], r)
            nm = m[:, i].copy()
            for lst in idx:
                nm[lst] = 1.0
            cols.append(nm)
        return {"pred_masks": np.stack(cols, 1), "pred_scores": s, "pred_classes": c}
    if name.startswith("msc"):
        # per-class min size: msc<c1>_<c2>
        n1, n2 = (int(x) for x in name[3:].split("_"))
        keep = np.array([sizes[i] >= (n1 if int(c[i]) == 1 else n2) for i in range(len(s))])
        return {"pred_masks": m[:, keep], "pred_scores": s[keep], "pred_classes": c[keep]}
    if name.startswith("cnms"):
        # class-aware NMS: suppress only same-class overlaps
        th = float(name[len("cnms"):])
        order = np.argsort(-s)
        kept = []
        drop = np.zeros(len(s), bool)
        for i in order:
            if sizes[i] == 0:
                drop[i] = True
                continue
            mi = mb[:, i]
            for j in kept:
                if c[j] != c[i]:
                    continue
                inter = np.logical_and(mi, mb[:, j]).sum()
                if inter / max(sizes[i] + sizes[j] - inter, 1) > th:
                    drop[i] = True
                    break
            if not drop[i]:
                kept.append(i)
        keep = ~drop
        return {"pred_masks": m[:, keep], "pred_scores": s[keep], "pred_classes": c[keep]}
    if name.startswith("nms"):
        th = float(name[len("nms"):])
        order = np.argsort(-s)
        kept = []
        drop = np.zeros(len(s), bool)
        for i in order:
            if sizes[i] == 0:
                drop[i] = True
                continue
            mi = mb[:, i]
            for j in kept:
                inter = np.logical_and(mi, mb[:, j]).sum()
                if inter / max(sizes[i] + sizes[j] - inter, 1) > th:
                    drop[i] = True
                    break
            if not drop[i]:
                kept.append(i)
        keep = ~drop
        return {"pred_masks": m[:, keep], "pred_scores": s[keep], "pred_classes": c[keep]}
    if name.startswith("dbc"):
        # per-class split eps: dbc<eps_class1>_<eps_class2>
        e1, e2 = (float(x) for x in name[3:].split("_"))
        xyz = xyz_of(scene)
        cols, scs, cls = [], [], []
        for i in range(m.shape[1]):
            mi = mb[:, i]
            n = int(mi.sum())
            if n == 0:
                continue
            eps = e1 if int(c[i]) == 1 else e2
            if n > 20000:
                cols.append(m[:, i]); scs.append(s[i]); cls.append(c[i]); continue
            lab = DBSCAN(eps=eps, min_samples=1, n_jobs=4).fit(xyz[mi]).labels_
            idx = np.where(mi)[0]
            for u in np.unique(lab):
                nm = np.zeros(m.shape[0], np.float32)
                nm[idx[lab == u]] = 1.0
                cols.append(nm); scs.append(s[i]); cls.append(c[i])
        return {"pred_masks": np.stack(cols, 1), "pred_scores": np.array(scs, np.float32),
                "pred_classes": np.array(cls, np.int64)}
    if name.startswith("dbl") or name.startswith("dbs"):
        # dbl<eps>: keep largest component; dbs<eps>: split into components
        split = name.startswith("dbs")
        eps = float(name[3:])
        xyz = xyz_of(scene)
        cols, scs, cls = [], [], []
        for i in range(m.shape[1]):
            mi = mb[:, i]
            n = int(mi.sum())
            if n == 0:
                continue
            if n > 20000:  # huge mask: certainly not an interaction element, keep as is
                cols.append(m[:, i]); scs.append(s[i]); cls.append(c[i]); continue
            lab = DBSCAN(eps=eps, min_samples=1, n_jobs=4).fit(xyz[mi]).labels_
            uniq, cnt = np.unique(lab, return_counts=True)
            if len(uniq) == 1:
                cols.append(m[:, i]); scs.append(s[i]); cls.append(c[i]); continue
            if split:
                idx = np.where(mi)[0]
                for u in uniq:
                    nm = np.zeros(m.shape[0], np.float32)
                    nm[idx[lab == u]] = 1.0
                    cols.append(nm); scs.append(s[i]); cls.append(c[i])
            else:
                u = uniq[np.argmax(cnt)]
                idx = np.where(mi)[0]
                nm = np.zeros(m.shape[0], np.float32)
                nm[idx[lab == u]] = 1.0
                cols.append(nm); scs.append(s[i]); cls.append(c[i])
        return {"pred_masks": np.stack(cols, 1), "pred_scores": np.array(scs, np.float32),
                "pred_classes": np.array(cls, np.int64)}
    raise ValueError(name)


def variant_chain(d, scene, name):
    for part in name.split("+"):
        if part != "base":
            d = variant_scene(d, scene, part)
    return d


def run(preds, name):
    v = {sc: variant_chain(d, sc, name) for sc, d in preds.items()}
    os.makedirs(OUT, exist_ok=True)
    ap50 = evaluate(v, GT, os.path.join(OUT, f"pred_{name}"), dataset="articulate3d",
                    eval_articulation=False, gt_articulations={}, eval_hierarchy_inter=False)
    print(f"VARIANT {name}: M_ap50 = {ap50}", flush=True)
    return ap50


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="/data/challenge/USDNet/results/inference_inter_seg_val/debug/val_preds/preds.pkl")
    ap.add_argument("--variants", nargs="*",
                    default=["base", "nms0.75", "nms0.5", "minsize5",
                             "dbl0.1", "dbl0.2", "dbs0.1", "dbs0.2"])
    args = ap.parse_args()
    preds = pickle.load(open(args.pkl, "rb"))
    for name in args.variants:
        try:
            run(preds, name)
        except Exception as e:
            print(f"VARIANT {name}: FAILED {type(e).__name__}: {e}", flush=True)
