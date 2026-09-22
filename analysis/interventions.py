"""Quantify cheap post-hoc fixes suggested by the error diagnosis, with exact benchmark AP.

Interventions operate ONLY on predicted quantities (no GT leakage):
  A. gravity snap  : rotation preds whose axis is within th of world +z -> set axis = z
  B. horizontal snap: translation preds whose axis is within th of the xy-plane -> zero the z comp
  C. origin de-shrink: rotation origins pushed away from the *predicted* mask centroid by (1+alpha)
  D. origin snap-to-mask: rotation origin replaced by nearest point of its own predicted mask
"""
import json
import os
import pickle
import numpy as np
import h5py

import benchmark.evaluate_semantic_instance as esi
from build_matches import set_articulate3d_labels, load_gt_articulation, GT_TXT_DIR, PRED_PKL

NPY_DIR = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_from(preds_mod):
    matches = {}
    for scene, pred in sorted(preds_mod.items()):
        gt_file = os.path.join(GT_TXT_DIR, scene + ".txt")
        arti = {np.int64(k): v for k, v in load_gt_articulation(scene).items()}
        gt2pred, pred2gt = esi.assign_instances_for_scan(
            pred, gt_file, True, {"articulations_dict": arti}, False)
        matches[os.path.abspath(gt_file)] = {"gt": gt2pred, "pred": pred2gt, "scene": scene}
    return matches


def aps(matches):
    def ap(c):
        a = esi.compute_averages(esi.evaluate_matches(matches, c))
        return float(a["all_ap_50%"]), float(a["classes"]["rotation"]["ap50%"]), float(a["classes"]["translation"]["ap50%"])
    return {
        "AP50": ap(esi.match_criteria_M_pred),
        "AP50_axis": ap(esi.match_criteria_MA_pred),
        "AP50_origin": ap(esi.match_criteria_MO_pred),
        "AP50_axis_origin": ap(esi.match_criteria_MAO_pred_standard),
    }


def main():
    set_articulate3d_labels()
    base = pickle.load(open(PRED_PKL, "rb"))
    xyz = {s: np.load(f"{NPY_DIR}/{s}.npy")[:, :3].astype(np.float64) for s in base}
    # predicted mask centroids, once
    cents = {}
    for s, p in base.items():
        m = p["pred_masks"] > 1e-3
        c = np.zeros((m.shape[1], 3))
        for i in range(m.shape[1]):
            if m[:, i].sum() > 0:
                c[i] = xyz[s][m[:, i]].mean(0)
        cents[s] = c

    def copy_preds():
        return {s: {k: (v.copy() if hasattr(v, "copy") else v) for k, v in p.items()} for s, p in base.items()}

    results = {}
    results["baseline"] = aps(build_from(base))
    print("baseline:", json.dumps(results["baseline"]))

    z = np.array([0, 0, 1.0])
    # A. gravity snap for rotation
    for th in [10, 20, 30, 45]:
        pm = copy_preds()
        n = 0
        for s, p in pm.items():
            ax = p["pred_axises"].astype(np.float64)
            for i in range(ax.shape[0]):
                if p["pred_classes"][i] == 1:
                    a = unit(ax[i])
                    if np.abs(a @ z) > np.cos(np.deg2rad(th)):
                        ax[i] = z * np.sign(a @ z)
                        n += 1
            p["pred_axises"] = ax.astype(np.float32)
        results[f"A_gravity_snap_rot_{th}deg"] = aps(build_from(pm))
        results[f"A_gravity_snap_rot_{th}deg"]["n_modified"] = n
        print(f"A gravity snap rot {th}deg (n={n}):", json.dumps(results[f"A_gravity_snap_rot_{th}deg"]))

    # B. horizontal snap for translation
    for th in [10, 20, 30]:
        pm = copy_preds()
        n = 0
        for s, p in pm.items():
            ax = p["pred_axises"].astype(np.float64)
            for i in range(ax.shape[0]):
                if p["pred_classes"][i] == 2:
                    a = unit(ax[i])
                    if np.abs(a[2]) < np.sin(np.deg2rad(th)):
                        b = np.array([a[0], a[1], 0.0])
                        if np.linalg.norm(b) > 1e-6:
                            ax[i] = unit(b)
                            n += 1
            p["pred_axises"] = ax.astype(np.float32)
        results[f"B_horiz_snap_trans_{th}deg"] = aps(build_from(pm))
        results[f"B_horiz_snap_trans_{th}deg"]["n_modified"] = n
        print(f"B horiz snap trans {th}deg (n={n}):", json.dumps(results[f"B_horiz_snap_trans_{th}deg"]))

    # C. origin de-shrink away from predicted mask centroid (rotation only)
    for alpha in [0.1, 0.2, 0.3, 0.5]:
        pm = copy_preds()
        for s, p in pm.items():
            o = p["pred_origins"].astype(np.float64)
            for i in range(o.shape[0]):
                if p["pred_classes"][i] == 1:
                    c = cents[s][i]
                    if np.any(c):
                        o[i] = c + (1 + alpha) * (o[i] - c)
            p["pred_origins"] = o.astype(np.float32)
        results[f"C_deshrink_{alpha}"] = aps(build_from(pm))
        print(f"C de-shrink alpha={alpha}:", json.dumps(results[f"C_deshrink_{alpha}"]))

    # D. snap origin to nearest point of its own predicted mask (rotation only)
    pm = copy_preds()
    for s, p in pm.items():
        o = p["pred_origins"].astype(np.float64)
        m = base[s]["pred_masks"] > 1e-3
        for i in range(o.shape[0]):
            if p["pred_classes"][i] == 1 and m[:, i].sum() > 0:
                pts = xyz[s][m[:, i]]
                o[i] = pts[np.argmin(((pts - o[i]) ** 2).sum(1))]
        p["pred_origins"] = o.astype(np.float32)
    results["D_snap_origin_to_mask"] = aps(build_from(pm))
    print("D snap origin to mask:", json.dumps(results["D_snap_origin_to_mask"]))

    # E. best combo: gravity snap 30 + de-shrink 0.2
    pm = copy_preds()
    for s, p in pm.items():
        ax = p["pred_axises"].astype(np.float64)
        o = p["pred_origins"].astype(np.float64)
        for i in range(ax.shape[0]):
            if p["pred_classes"][i] == 1:
                a = unit(ax[i])
                if np.abs(a @ z) > np.cos(np.deg2rad(30)):
                    ax[i] = z * np.sign(a @ z)
                c = cents[s][i]
                if np.any(c):
                    o[i] = c + 1.2 * (o[i] - c)
        p["pred_axises"] = ax.astype(np.float32)
        p["pred_origins"] = o.astype(np.float32)
    results["E_combo_gravity30_deshrink0.2"] = aps(build_from(pm))
    print("E combo:", json.dumps(results["E_combo_gravity30_deshrink0.2"]))

    json.dump(results, open("/data/challenge/analysis/arti_interventions.json", "w"), indent=1)
    print("wrote /data/challenge/analysis/arti_interventions.json")


if __name__ == "__main__":
    main()
