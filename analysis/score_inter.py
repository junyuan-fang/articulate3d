#!/usr/bin/env python
"""Score an interactable-track preds.pkl with the official evaluator.

The inter phase reports plain AP50 over interactable-part masks. Unlike the
movable track, these predictions carry no axis/origin, so assign_instances_for_scan
must be called with eval_articulation=False.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "/data/challenge/USDNet")
sys.path.insert(0, "/data/challenge/analysis")

import benchmark.evaluate_semantic_instance as esi
import build_matches as bm

GT_TXT = "/data/challenge/USDNet/data/processed/articulate3d_challenge_inter/instance_gt/validation"


def build_inter(pred_pkl):
    bm.set_articulate3d_labels()
    preds = pickle.load(open(pred_pkl, "rb"))
    matches = {}
    for scene, pred in sorted(preds.items()):
        gt_file = os.path.join(GT_TXT, scene + ".txt")
        if not os.path.isfile(gt_file):
            continue
        gt2pred, pred2gt = esi.assign_instances_for_scan(pred, gt_file, False, None, False)
        matches[os.path.abspath(gt_file)] = {"gt": gt2pred, "pred": pred2gt, "scene": scene}
    return matches


def main():
    pred_pkl = sys.argv[1]
    matches = build_inter(pred_pkl)
    ap = esi.compute_averages(esi.evaluate_matches(matches, esi.match_criteria_M_pred))
    print(f"  {os.path.basename(pred_pkl):26s} scenes={len(matches)}  "
          f"AP50={ap['all_ap_50%']:.4f}  AP={ap['all_ap']:.4f}  AP25={ap['all_ap_25%']:.4f}")


if __name__ == "__main__":
    main()
