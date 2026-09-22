#!/usr/bin/env python
"""Why does a "perfect GT mask" oracle NOT score 1.0 on Articulate3D val?

Everything here is measured against the *unmodified* official evaluator
(`USDNet/benchmark/evaluate_semantic_instance.py`).  The only patched evaluator
is a generated COPY at `analysis/_evaluate_semantic_instance_clipped.py`
(created by this script) used for the STEP-3 intervention.

Sections
  0  GT bookkeeping                      -> H2, H4, H7
  1  four oracles (a)-(d) + model baseline
  2  H1 instrumentation (arccos NaN), split by evaluator call direction
  3  H5 PR-curve mechanics, by hand
  4  STEP 3 intervention: np.clip patch on a copy of the evaluator
  5  sensitivity: axis float representation / decimal precision
  6  alternative oracle constructions (query-slot based)

Usage:
    cd /data/challenge/analysis
    PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis \
    $HOME/.conda/envs/usdnet/bin/python -u oracle_gap.py
"""
import os
import sys
import glob
import json
import pickle
import importlib.util
from collections import Counter

import numpy as np

import build_matches as bm
import benchmark.evaluate_semantic_instance as esi
import benchmark.util_3d as util_3d

OUT_JSON = "/data/challenge/analysis/oracle_gap_results.json"
PRED_PKL = ("/data/challenge/USDNet/results/eval_mov_val_einsum/debug/"
            "val_preds/preds.pkl")
POINTS_DIR = ("/data/challenge/USDNet/data/processed/"
              "articulate3d_challenge_mov/validation")
VALID = {1, 2}

CRITERIA = [
    ("AP50", "match_criteria_M_pred"),
    ("AP50_axis", "match_criteria_MA_pred"),
    ("AP50_origin", "match_criteria_MO_pred"),
    ("AP50_axis_origin", "match_criteria_MAO_pred_standard"),   # = MAO_ST
]


# --------------------------------------------------------------------------- #
def scenes():
    return sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(bm.GT_TXT_DIR + "/*.txt"))


def gt_ids_of(scene):
    return util_3d.load_ids(os.path.join(bm.GT_TXT_DIR, scene + ".txt"))


def gt_arti_of(scene):
    return bm.load_gt_articulation(scene)


def build_matches(preds, mod=esi):
    matches = {}
    for scene, pred in sorted(preds.items()):
        gt_file = os.path.join(bm.GT_TXT_DIR, scene + ".txt")
        arti = {np.int64(k): v for k, v in gt_arti_of(scene).items()}
        gt2pred, pred2gt = mod.assign_instances_for_scan(
            pred, gt_file, True, {"articulations_dict": arti}, False)
        matches[os.path.abspath(gt_file)] = {"gt": gt2pred, "pred": pred2gt,
                                             "scene": scene}
    return matches


def score(matches, mod=esi):
    res = {}
    for name, attr in CRITERIA:
        ap = mod.compute_averages(mod.evaluate_matches(matches, getattr(mod, attr)))
        res[name] = float(ap["all_ap_50%"])
        res[name + "_mAP"] = float(ap["all_ap"])
        for c in mod.CLASS_LABELS:
            res[f"{name}::{c}"] = float(ap["classes"][c]["ap50%"])
    return res


def brief(r):
    return "  ".join(f"{k}={r[k]:.4f}" for k, _ in CRITERIA)


# --------------------------------------------------------------------------- #
#  oracle builders
# --------------------------------------------------------------------------- #
def axis_repr(a, mode):
    a = np.asarray(a, np.float64)
    if mode == "raw":
        return a.copy()
    if mode == "norm64":
        return a / np.linalg.norm(a)
    if mode == "f32":
        return a.astype(np.float32).astype(np.float64)
    if mode == "f16":
        return a.astype(np.float16).astype(np.float64)
    if mode.startswith("dec"):
        p = int(mode[3:])
        return np.array([float(("%%.%df" % p) % x) for x in a])
    raise ValueError(mode)


def build_gt_oracle(conf_mode="ones", rng=None, axmode="raw",
                    arti_source="gt", pred_arti=None, ax_dtype=np.float64):
    """(a)/(b)/(c): exactly one prediction per GT instance the evaluator sees.
    Mask = the GT point set, class = instance_id//1000."""
    out = {}
    for s in scenes():
        ids = gt_ids_of(s)
        arti = gt_arti_of(s)
        keep = [int(i) for i in np.unique(ids) if i != 0 and (i // 1000) in VALID]
        n = len(keep)
        masks = np.zeros((len(ids), n), np.float32)
        cls = np.zeros(n, np.int64)
        ax = np.zeros((n, 3), ax_dtype)
        og = np.zeros((n, 3), np.float64)
        for j, i in enumerate(keep):
            masks[:, j] = (ids == i).astype(np.float32)
            cls[j] = i // 1000
            ax[j] = axis_repr(arti[i]["axis"], axmode)
            og[j] = arti[i]["origin"]
        if conf_mode == "ones":
            sc = np.ones(n, np.float32)
        elif conf_mode == "random":
            sc = rng.random(n).astype(np.float32)
        elif conf_mode == "descending":
            sc = np.linspace(1.0, 0.01, max(n, 1))[:n].astype(np.float32)
        else:
            raise ValueError(conf_mode)
        if arti_source == "pred" and s in pred_arti:
            p = pred_arti[s]
            pm = np.asarray(p["pred_masks"]) > 1e-3
            for j, i in enumerate(keep):
                g = ids == i
                inter = (pm & g[:, None]).sum(0)
                iou = inter / np.maximum(g.sum() + pm.sum(0) - inter, 1)
                q = int(np.argmax(iou))
                ax[j] = p["pred_axises"][q]
                og[j] = p["pred_origins"][q]
        out[s] = dict(pred_masks=masks, pred_classes=cls, pred_scores=sc,
                      pred_axises=ax, pred_origins=og)
    return out


def build_pred_masks_gt_arti(raw):
    """(d) model masks; axis/origin of every query that matches a GT at
    IoU>0.5 replaced by that GT's axis/origin."""
    out, n_rep = {}, 0
    for s, p in raw.items():
        ids = gt_ids_of(s)
        arti = gt_arti_of(s)
        pm = np.asarray(p["pred_masks"]) > 1e-3
        ax = np.asarray(p["pred_axises"], np.float64).copy()
        og = np.asarray(p["pred_origins"], np.float64).copy()
        for gid, a in arti.items():
            g = ids == gid
            if g.sum() == 0:
                continue
            inter = (pm & g[:, None]).sum(0)
            iou = inter / np.maximum(g.sum() + pm.sum(0) - inter, 1)
            q = int(np.argmax(iou))
            if iou[q] > 0.5:
                ax[q] = a["axis"]
                og[q] = a["origin"]
                n_rep += 1
        out[s] = dict(p, pred_axises=ax, pred_origins=og)
    return out, n_rep


def build_queryslot_oracle(raw, axmode="raw", keep_other=False, conf="ones"):
    """Alternative construction: write the GT into the best-IoU *query slot*.
    When two GT instances pick the same slot one of them silently loses its
    prediction (a hard FN with no accompanying FP)."""
    out, stat = {}, dict(n_pred=0, n_gt=0, gt_no_pred=0, collisions=0)
    for s in scenes():
        ids = gt_ids_of(s)
        arti = gt_arti_of(s)
        keep = [int(i) for i in np.unique(ids) if i != 0 and (i // 1000) in VALID]
        stat["n_gt"] += len(keep)
        if s not in raw:
            continue
        p = raw[s]
        pm = np.asarray(p["pred_masks"]) > 1e-3
        gt_of_slot = {}
        for i in keep:
            g = ids == i
            inter = (pm & g[:, None]).sum(0)
            iou = inter / np.maximum(g.sum() + pm.sum(0) - inter, 1)
            q = int(np.argmax(iou))
            if q in gt_of_slot:
                stat["collisions"] += 1
            gt_of_slot[q] = i
        stat["gt_no_pred"] += len(keep) - len(set(gt_of_slot.values()))
        slots = list(range(pm.shape[1])) if keep_other else sorted(gt_of_slot)
        n = len(slots)
        masks = np.zeros((len(ids), n), np.float32)
        cls = np.zeros(n, np.int64)
        ax = np.zeros((n, 3)); og = np.zeros((n, 3))
        sc = np.ones(n, np.float32)
        for j, q in enumerate(slots):
            if q in gt_of_slot:
                i = gt_of_slot[q]
                masks[:, j] = (ids == i).astype(np.float32)
                cls[j] = i // 1000
                ax[j] = axis_repr(arti[i]["axis"], axmode)
                og[j] = arti[i]["origin"]
                if conf == "model":
                    sc[j] = p["pred_scores"][q]
            else:
                masks[:, j] = np.asarray(p["pred_masks"][:, q], np.float32)
                cls[j] = p["pred_classes"][q]
                ax[j] = p["pred_axises"][q]
                og[j] = p["pred_origins"][q]
                sc[j] = p["pred_scores"][q]
        stat["n_pred"] += n
        out[s] = dict(pred_masks=masks, pred_classes=cls, pred_scores=sc,
                      pred_axises=ax, pred_origins=og)
    return out, stat


# --------------------------------------------------------------------------- #
#  H1 instrumentation: |cos| > 1 -> arccos = nan -> perfect axis scored MISS
# --------------------------------------------------------------------------- #
class AxisProbe:
    """Drop-in replacement for esi.match_criteria_MA_pred that is
    line-for-line identical but records what happens."""

    def reset(self):
        self.n_calls = 0
        self.nan = Counter()          # (gt_first, label) -> count
        self.nan_iou_ok = Counter()
        self.max_dev = 0.0

    def install(self, mod=esi):
        self.reset()
        self._orig = mod.match_criteria_MA_pred
        probe = self

        def wrapped(inst_ref, inst_check, iou_th, gt_first=False):
            axis_th = 15
            overlap = float(inst_ref["intersection"]) / (
                inst_check["vert_count"] + inst_ref["vert_count"]
                - inst_ref["intersection"])
            is_iou = overlap > iou_th
            a, b = inst_ref["axis"], inst_check["axis"]
            dev = np.abs(np.dot(a, b) /
                         (np.linalg.norm(a) * np.linalg.norm(b)))
            probe.n_calls += 1
            probe.max_dev = max(probe.max_dev, float(dev))
            ang = np.arccos(dev)
            if np.isnan(ang):
                lab = mod.ID_TO_LABEL[int(inst_ref["label_id"])]
                probe.nan[(bool(gt_first), lab)] += 1
                if is_iou:
                    probe.nan_iou_ok[(bool(gt_first), lab)] += 1
            return is_iou and (ang < np.deg2rad(axis_th))

        mod.match_criteria_MA_pred = wrapped

    def uninstall(self, mod=esi):
        mod.match_criteria_MA_pred = self._orig

    def report(self, n_overlaps=10, n_evals=2):
        """Each (instance, overlap threshold, criterion) triggers one call per
        direction, so divide by n_overlaps * n_evals to get instance counts."""
        d = n_overlaps * n_evals
        gt_loop = {k[1]: v / d for k, v in self.nan_iou_ok.items() if not k[0]}
        pr_loop = {k[1]: v / d for k, v in self.nan_iou_ok.items() if k[0]}
        return dict(n_calls=self.n_calls, max_dev=self.max_dev,
                    instances_nan_gt_loop_FN=gt_loop,
                    instances_nan_pred_loop_FP=pr_loop)


# --------------------------------------------------------------------------- #
def ap_by_hand(y_true, y_score, hard_fn):
    """Verbatim re-implementation of the evaluator's PR integration (H5)."""
    y_true = np.asarray(y_true, float); y_score = np.asarray(y_score, float)
    o = np.argsort(y_score); ys, yt = y_score[o], y_true[o]
    cs = np.cumsum(yt)
    _, ui = np.unique(ys, return_index=True)
    npr, ne = len(ui) + 1, len(ys)
    nte = cs[-1] if len(cs) else 0
    prec, rec = np.zeros(npr), np.zeros(npr)
    cs = np.append(cs, 0)
    for ir, isc in enumerate(ui):
        cum = cs[isc - 1]
        tp = nte - cum; fp = ne - isc - tp; fn = cum + hard_fn
        prec[ir] = tp / (tp + fp); rec[ir] = tp / (tp + fn)
    prec[-1], rec[-1] = 1.0, 0.0
    rc = np.append(np.append(rec[0], rec), 0.0)
    return float(np.dot(prec, np.convolve(rc, [-0.5, 0, 0.5], "valid")))


# --------------------------------------------------------------------------- #
def main():
    bm.set_articulate3d_labels()
    R = {}
    scs = scenes()
    print(f"validation scenes: {len(scs)}", flush=True)

    # ---- 0. GT bookkeeping: H2 / H4 / H7 ------------------------------- #
    st = dict(n_scenes=len(scs), gt_txt_ids=0, gt_valid=0, gt_invalid_class=0,
              gt_id_lt_1000=0, gt_vert_lt_min_region=0, gt_missing_h5=0,
              h5_not_in_txt=0, axis_zero=0, axis_nan=0, origin_nan=0,
              axis_nonunit=0, per_class=Counter(), empty_scenes=[],
              mask_space_mismatch=0, min_gt_verts={},
              exactly_axis_aligned=Counter())
    norms = []
    for s in scs:
        ids = gt_ids_of(s)
        arti = gt_arti_of(s)
        npts = np.load(os.path.join(POINTS_DIR, s + ".npy"), mmap_mode="r").shape[0]
        if npts != len(ids):
            st["mask_space_mismatch"] += 1
        uniq = [int(i) for i in np.unique(ids) if i != 0]
        st["gt_txt_ids"] += len(uniq)
        nv = 0
        for i in uniq:
            lab = i // 1000
            if lab not in VALID:
                st["gt_invalid_class"] += 1
                continue
            if i < 1000:
                st["gt_id_lt_1000"] += 1
            if int((ids == i).sum()) < esi.opt["min_region_sizes"][0]:
                st["gt_vert_lt_min_region"] += 1
            st["gt_valid"] += 1; nv += 1
            st["per_class"][esi.ID_TO_LABEL[lab]] += 1
            if i not in arti:
                st["gt_missing_h5"] += 1
            else:
                v = np.asarray(arti[i]["axis"], np.float64)
                if (np.abs(v) == 1.0).sum() == 1 and (v == 0).sum() == 2:
                    st["exactly_axis_aligned"][esi.ID_TO_LABEL[lab]] += 1
        if nv == 0:
            st["empty_scenes"].append(s)
        for k, a in arti.items():
            if (ids == k).sum() == 0:
                st["h5_not_in_txt"] += 1
            n = float(np.linalg.norm(a["axis"]))
            norms.append(n)
            st["axis_zero"] += n < 1e-8
            st["axis_nonunit"] += abs(n - 1.0) > 1e-6
            st["axis_nan"] += not np.isfinite(a["axis"]).all()
            st["origin_nan"] += not np.isfinite(a["origin"]).all()
    st["per_class"] = dict(st["per_class"])
    st["exactly_axis_aligned"] = dict(st["exactly_axis_aligned"])
    st["axis_norm_range"] = [float(min(norms)), float(max(norms))]
    st["axis_norm_exactly_1"] = int((np.asarray(norms) == 1.0).sum())
    R["gt_stats"] = st
    print(json.dumps(st, indent=2, default=str), flush=True)

    # ---- 1+2. oracles (a)/(b) with H1 probe ---------------------------- #
    probe = AxisProbe()
    for tag, kw in [("a_gt_all_conf1", dict(conf_mode="ones")),
                    ("b_gt_all_conf_random",
                     dict(conf_mode="random", rng=np.random.default_rng(0))),
                    ("b2_gt_all_conf_descending", dict(conf_mode="descending"))]:
        preds = build_gt_oracle(**kw)
        m = build_matches(preds)
        probe.install()
        r = score(m)
        probe.uninstall()
        r["_probe"] = probe.report()
        R[tag] = r
        print(f"\n[{tag}] {brief(r)}", flush=True)
        print("   per-class MAO_ST: rot=%.4f tr=%.4f"
              % (r["AP50_axis_origin::rotation"],
                 r["AP50_axis_origin::translation"]), flush=True)
        print("   H1 probe:", r["_probe"], flush=True)
        del preds, m

    # ---- 3. H5 by hand -------------------------------------------------- #
    R["h5_pr_mechanics"] = {
        "100 TP, one score, no FN/FP": ap_by_hand(np.ones(100), np.ones(100), 0),
        "100 TP, random scores": ap_by_hand(
            np.ones(100), np.random.default_rng(1).random(100), 0),
        "99 TP + 1 hard FN, one score": ap_by_hand(np.ones(99), np.ones(99), 1),
        "99 TP + 1 FP, one score": ap_by_hand(
            np.append(np.ones(99), 0), np.ones(100), 0),
        "99 TP + 1 FP + 1 hard FN": ap_by_hand(
            np.append(np.ones(99), 0), np.ones(100), 1),
    }
    print("\nH5:", R["h5_pr_mechanics"], flush=True)

    # ---- 4. STEP 3 intervention on a COPY of the evaluator -------------- #
    patched = "/data/challenge/analysis/_evaluate_semantic_instance_clipped.py"
    src = open("/data/challenge/USDNet/benchmark/"
               "evaluate_semantic_instance.py").read()
    old = "    deriavation = np.abs(deriavation)\n"
    assert src.count(old) == 1
    open(patched, "w").write(src.replace(
        old, "    deriavation = np.clip(np.abs(deriavation), -1.0, 1.0)\n"))
    spec = importlib.util.spec_from_file_location("esi_clipped", patched)
    esi_c = importlib.util.module_from_spec(spec)
    sys.modules["esi_clipped"] = esi_c
    spec.loader.exec_module(esi_c)
    esi_c.CLASS_LABELS = ("rotation", "translation")
    esi_c.VALID_CLASS_IDS = np.array([1, 2])
    esi_c.ID_TO_LABEL = {1: "rotation", 2: "translation"}
    esi_c.LABEL_TO_ID = {"rotation": 1, "translation": 2}

    preds = build_gt_oracle(conf_mode="ones")
    R["a_gt_all_conf1_CLIPPED"] = score(build_matches(preds, mod=esi_c), mod=esi_c)
    print("\n[a + np.clip patch (copy of evaluator)] "
          + brief(R["a_gt_all_conf1_CLIPPED"]), flush=True)
    del preds

    # ---- 5. sensitivity to the float representation of the copied axis --- #
    sens = {}
    for axmode, dt in ([(m, np.float64) for m in
                        ["raw", "norm64", "f32", "f16"] +
                        [f"dec{p}" for p in range(3, 18)]]
                       + [("raw", np.float32)]):
        key = axmode + ("" if dt is np.float64 else "_stored_f32")
        preds = build_gt_oracle(conf_mode="ones", axmode=axmode, ax_dtype=dt)
        m = build_matches(preds)
        probe.install()
        r = score(m)
        probe.uninstall()
        sens[key] = dict(MAO_ST=r["AP50_axis_origin"], AP50=r["AP50"],
                         rot=r["AP50_axis_origin::rotation"],
                         tr=r["AP50_axis_origin::translation"],
                         probe=probe.report())
        print(f"  axis={key:16s} AP50={r['AP50']:.4f} MAO_ST={r['AP50_axis_origin']:.4f}"
              f"  rot={r['AP50_axis_origin::rotation']:.4f}"
              f"  tr={r['AP50_axis_origin::translation']:.4f}"
              f"  nanFN={probe.report()['instances_nan_gt_loop_FN']}", flush=True)
        del preds, m
    R["axis_representation_sensitivity"] = sens

    # ---- 6. oracles (c)/(d), query-slot oracles, model baseline --------- #
    print("\nloading preds.pkl ...", flush=True)
    raw = pickle.load(open(PRED_PKL, "rb"))
    print(f"  {len(raw)} scenes", flush=True)

    preds = build_gt_oracle(conf_mode="ones", arti_source="pred", pred_arti=raw)
    R["c_gtmask_gtclass_predarti"] = score(build_matches(preds))
    print("[c GT mask+class, PREDICTED axis/origin] "
          + brief(R["c_gtmask_gtclass_predarti"]), flush=True)
    del preds

    preds, nrep = build_pred_masks_gt_arti(raw)
    R["d_predmask_gtarti"] = score(build_matches(preds))
    R["d_predmask_gtarti"]["_n_replaced"] = nrep
    print(f"[d PREDICTED mask, GT axis/origin] (n_replaced={nrep}) "
          + brief(R["d_predmask_gtarti"]), flush=True)
    del preds

    R["model_baseline"] = score(build_matches(raw))
    print("[model baseline] " + brief(R["model_baseline"]), flush=True)

    qs = {}
    for axmode in ["raw", "dec6", "f16"]:
        for ko in [False, True]:
            preds, stat = build_queryslot_oracle(raw, axmode, ko)
            r = score(build_matches(preds))
            qs[f"{axmode}_keepother{int(ko)}"] = dict(
                AP50=r["AP50"], MAO_ST=r["AP50_axis_origin"],
                rot=r["AP50_axis_origin::rotation"],
                tr=r["AP50_axis_origin::translation"], **stat)
            print(f"  queryslot axis={axmode:5s} keep_other={int(ko)} "
                  f"AP50={r['AP50']:.4f} MAO_ST={r['AP50_axis_origin']:.4f} "
                  f"gt_no_pred={stat['gt_no_pred']}", flush=True)
            del preds
    R["queryslot_oracles"] = qs

    json.dump(R, open(OUT_JSON, "w"), indent=2, default=str)
    print("\nwrote", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
