"""Consolidate all diagnosis outputs into one machine-readable summary."""
import json

D = json.load(open("/data/challenge/analysis/arti_error_diagnosis.json"))
F = json.load(open("/data/challenge/analysis/arti_bias_followup.json"))
I = json.load(open("/data/challenge/analysis/arti_interventions.json"))
C = json.load(open("/data/challenge/analysis/arti_nan_combo.json"))

S = {
    "what": "USDNet baseline articulation (axis/origin) error diagnosis, Articulate3D val",
    "reproduce": "source /data/challenge/USDNet/env.sh && cd /data/challenge/analysis && "
                 "PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis python diagnose_arti_errors.py "
                 "&& python followup_bias.py && python interventions.py && python nan_and_combo.py",
    "official_ap_reproduced": D["official"],
    "coverage": dict(D["coverage"], scenes_total=42,
                     scene_skipped="a980334473 (0 GT instances, absent from preds.pkl -> no effect)",
                     gt_rotation=236, gt_translation=154, matched_rotation=180, matched_translation=95),
    "error_distributions": {k: D[k] for k in ["all", "rotation", "translation"]},
    "concentration": D["concentration"],
    "systematic_axis": D["systematic_axis"],
    "systematic_origin": D["systematic_origin"],
    "origin_bias_by_axis_orientation": F["origin_bias_by_axis_orientation"],
    "oracle_global_shifts": {k: F[k] for k in ["oracle_global_z_shift_horizontal_rot", "oracle_global_xyz_shift_all_rot"]},
    "near_orthogonal_cases": F["near_orthogonal_cases"],
    "failure_overlap_rotation": F["failure_overlap_rotation"],
    "trivial_baselines": F["baselines"],
    "baseline_win_tie_loss": {
        "axis_model_vs_zup_prior": {
            "rotation": {"n": 180, "model_pass": 0.8833, "zup_pass": 0.9333,
                         "win_model_only": 0, "loss_zup_only": 9, "tie_pass": 159, "tie_fail": 12},
            "translation": {"n": 95, "model_pass": 0.8421, "zup_pass": 0.0,
                            "win_model_only": 80, "loss_zup_only": 0, "tie_pass": 0, "tie_fail": 15},
            "all": {"n": 275, "model_pass": 0.8691, "zup_pass": 0.6109,
                    "win_model_only": 80, "loss_zup_only": 9, "tie_pass": 159, "tie_fail": 27},
        },
        "origin_model_vs_part_centroid": {"n": 180, "model_pass": 0.7278, "centroid_pass": 0.2778,
                                          "win": 86, "loss": 5, "tie_pass": 45, "tie_fail": 44},
    },
    "gt_axis_orientation_breakdown": {
        "rotation_vertical_gt_axis": {"n": 168, "axis_pass15": 0.946, "median_deg": 2.62},
        "rotation_horizontal_gt_axis": {"n": 8, "axis_pass15": 0.0,
                                        "errors_deg": [87.3, 49.0, 89.2, 87.7, 90.0, 87.5, 87.1, 85.5]},
        "translation_gt_axis_all_horizontal": {"n": 95, "frac_horizontal": 1.0,
                                               "axis_pass15": 0.842, "median_deg": 5.91},
    },
    "size_buckets": D["size_buckets"],
    "correlations": D["correlations"],
    "per_scene": D["per_scene"],
    "per_scene_medians": {"axis_deg_median_of_scene_medians": 3.30, "axis_min": 0.98, "axis_max": 19.34,
                          "origin_rot_median_of_scene_medians": 0.131, "origin_min": 0.007, "origin_max": 0.772,
                          "n_scenes": 41},
    "axis_fail_scene_concentration": D["axis_fail_scene_concentration"],
    "threshold_sweep_exact_ap": D["threshold_sweep"],
    "ap_gap_decomposition": F["ap_gap_decomposition"],
    "per_class_ap": F["per_class_ap"],
    "post_hoc_interventions_ap": I,
    "post_hoc_combo_ap": C,
    "nan_edge_case": {"gt_axes_triggering_arccos_nan_with_pure_z": 0,
                      "pred_gt_pairs_with_nan_baseline": 0, "pred_gt_pairs_with_nan_after_gravity_snap": 0,
                      "conclusion": "checked, immaterial; eps-tilted and exact-z snapping give identical AP"},
}
json.dump(S, open("/data/challenge/analysis/SUMMARY_arti_diagnosis.json", "w"), indent=1)
print("wrote /data/challenge/analysis/SUMMARY_arti_diagnosis.json")
