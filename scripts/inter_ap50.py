#!/usr/bin/env python
"""Local AP50 scorer + postprocess grid for the inter task.

Scoring follows the ScanNet-style evaluation the server uses: per semantic
class, predictions ranked by score, greedy-matched to GT at IoU>=0.5, AP =
area under the interpolated PR curve, then mean over classes present in GT.
Calibrate against the server's val AP50 = 0.5363 before trusting deltas.

Postprocess knobs (all physically motivated, applied per scene):
  --max-size N   drop predictions with more than N points (elements are tiny)
  --min-size N   drop predictions with fewer than N points
  --nms T        greedy NMS: drop lower-scored pred if IoU with a kept one > T
"""
import argparse, os, pickle, sys
import numpy as np

VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_inter/validation"
MASK_TH = 1e-3


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(preds, max_size=None, min_size=None, nms=None):
    # collect per-class detections and GT
    dets = {}   # cls -> list of (score, scene, mask_bool)
    ngt = {}    # cls -> count
    gts = {}    # (scene, cls) -> list of gt masks
    for scene in sorted(preds):
        p = os.path.join(VAL, scene + ".npy")
        if not os.path.exists(p):
            continue
        a = np.load(p, mmap_mode="r")
        sem = np.asarray(a[:, 9]).astype(int)
        inst = np.asarray(a[:, 10]).astype(int)
        for gi in np.unique(inst):
            gm = inst == gi
            lab = int(np.bincount(sem[gm]).argmax())
            if lab == 0 or gm.sum() < 10:
                continue
            gts.setdefault((scene, lab), []).append(gm)
            ngt[lab] = ngt.get(lab, 0) + 1
        pm = np.asarray(preds[scene]["pred_masks"]) > MASK_TH
        sc = np.asarray(preds[scene]["pred_scores"]).ravel()
        cl = np.asarray(preds[scene]["pred_classes"]).ravel().astype(int)
        sizes = pm.sum(0)
        keep = np.ones(pm.shape[1], bool)
        if max_size is not None:
            keep &= sizes <= max_size
        if min_size is not None:
            keep &= sizes >= min_size
        order = np.argsort(-sc)
        if nms is not None:
            kept_idx = []
            for i in order:
                if not keep[i]:
                    continue
                mi = pm[:, i]
                si = sizes[i]
                dup = False
                for j in kept_idx:
                    intersec = np.logical_and(mi, pm[:, j]).sum()
                    if intersec / max(si + sizes[j] - intersec, 1) > nms:
                        dup = True
                        break
                if dup:
                    keep[i] = False
                else:
                    kept_idx.append(i)
        for i in range(pm.shape[1]):
            if keep[i]:
                dets.setdefault(cl[i], []).append((float(sc[i]), scene, pm[:, i]))

    aps = []
    for c, n in ngt.items():
        d = sorted(dets.get(c, []), key=lambda t: -t[0])
        matched = {}
        tp = np.zeros(len(d)); fp = np.zeros(len(d))
        for k, (s, scene, m) in enumerate(d):
            best, bi = 0.0, -1
            for gi, gm in enumerate(gts.get((scene, c), [])):
                if matched.get((scene, gi)):
                    continue
                inter = np.logical_and(m, gm).sum()
                iou = inter / max(m.sum() + gm.sum() - inter, 1)
                if iou > best:
                    best, bi = iou, gi
            if best >= 0.5:
                matched[(scene, bi)] = True
                tp[k] = 1
            else:
                fp[k] = 1
        rec = np.cumsum(tp) / n
        prec = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1)
        aps.append(voc_ap(rec, prec))
    return float(np.mean(aps)), {c: n for c, n in ngt.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", nargs="?", default="/data/challenge/USDNet/results/inference_inter_seg_val/debug/val_preds/preds.pkl")
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    preds = pickle.load(open(args.pkl, "rb"))
    base, ngt = evaluate(preds)
    print(f"baseline (no postprocess): AP50 = {base:.4f}   GT per class: {ngt}", flush=True)
    if args.grid:
        for ms in [None, 500, 300, 200, 120]:
            for nm in [None, 0.6, 0.4, 0.25]:
                if ms is None and nm is None:
                    continue
                v, _ = evaluate(preds, max_size=ms, nms=nm)
                print(f"max_size={ms} nms={nm}: AP50 = {v:.4f}", flush=True)
