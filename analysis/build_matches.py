"""Build the exact `matches` structure the USDNet benchmark uses, from preds.pkl + GT.

Mirrors benchmark.evaluate_semantic_instance.evaluate() for dataset='articulate3d',
eval_articulation=True, so that evaluate_matches() on the result reproduces the
official val numbers bit-for-bit.
"""
import os
import pickle
import numpy as np
import h5py

import benchmark.evaluate_semantic_instance as esi

ROOT = "/data/challenge/USDNet"
PRED_PKL = f"{ROOT}/results/eval_mov_val/debug/val_preds/preds.pkl"
GT_TXT_DIR = f"{ROOT}/data/processed/articulate3d_challenge_mov/instance_gt/validation"
GT_H5_DIR = f"{ROOT}/data/processed/articulate3d_challenge_mov/validation"


def set_articulate3d_labels():
    esi.CLASS_LABELS = ("rotation", "translation")
    esi.VALID_CLASS_IDS = np.array([1, 2])
    esi.ID_TO_LABEL = {}
    esi.LABEL_TO_ID = {}
    for i in range(len(esi.VALID_CLASS_IDS)):
        esi.LABEL_TO_ID[esi.CLASS_LABELS[i]] = esi.VALID_CLASS_IDS[i]
        esi.ID_TO_LABEL[esi.VALID_CLASS_IDS[i]] = esi.CLASS_LABELS[i]


def load_gt_articulation(scene):
    """h5 keys are raw instance ids; benchmark keys are sem_id*1000 + raw_id + 1."""
    out = {}
    with h5py.File(f"{GT_H5_DIR}/{scene}_articulation.h5", "r") as f:
        for k in f.keys():
            g = f[k]
            sem = int(np.array(g["sem_id"]))
            key = int(k) + sem * 1000 + 1
            out[key] = {
                "axis": np.array(g["axis"]).astype(np.float64),
                "origin": np.array(g["origin"]).astype(np.float64),
                "sem_id": sem,
                "raw_id": int(k),
                "inter_mask_count": int(np.array(g["inter_mask"]).sum()),
            }
    return out


def build(scenes=None):
    set_articulate3d_labels()
    preds = pickle.load(open(PRED_PKL, "rb"))
    if scenes is not None:
        preds = {k: v for k, v in preds.items() if k in scenes}
    matches = {}
    gt_arti_all = {}
    for i, (scene, pred) in enumerate(sorted(preds.items())):
        gt_file = os.path.join(GT_TXT_DIR, scene + ".txt")
        assert os.path.isfile(gt_file), gt_file
        arti = load_gt_articulation(scene)
        gt_arti_all[scene] = arti
        # np.int64 keys -> the benchmark calls m.item(); emulate with a dict of np ints
        arti_dict = {np.int64(k): v for k, v in arti.items()}
        gt2pred, pred2gt = esi.assign_instances_for_scan(
            pred, gt_file, True, {"articulations_dict": arti_dict}, False
        )
        matches[os.path.abspath(gt_file)] = {"gt": gt2pred, "pred": pred2gt, "scene": scene}
    return matches, gt_arti_all, preds


if __name__ == "__main__":
    matches, gt_arti, preds = build()
    print("scenes:", len(matches))
    for name, crit in [
        ("AP50 (M)", esi.match_criteria_M_pred),
        ("AP50_axis (MA)", esi.match_criteria_MA_pred),
        ("AP50_origin (MO)", esi.match_criteria_MO_pred),
        ("AP50_axis_origin (MAO_ST)", esi.match_criteria_MAO_pred_standard),
    ]:
        ap = esi.compute_averages(esi.evaluate_matches(matches, crit))
        print(f"{name:28s} AP50={ap['all_ap_50%']:.4f}  AP={ap['all_ap']:.4f}")
