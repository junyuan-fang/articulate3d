"""Follow-up: direction of the origin bias, near-orthogonal axis cases, per-class AP,
axis/origin failure overlap, and trivial baselines. Reads rows from the main diagnosis JSON."""
import json
import numpy as np
import benchmark.evaluate_semantic_instance as esi
from build_matches import build
from diagnose_arti_errors import make_MA, make_MO, make_MAO_ST, unit, axis_angle_deg, perp_dist, q

R = json.load(open("/data/challenge/analysis/arti_error_diagnosis.json"))
rows = R["rows"]
out = {}

A = np.array([r["axis_deg"] for r in rows])
O = np.array([r["origin_perp"] for r in rows])
rot = np.array([r["label"] == "rotation" for r in rows])
gt_ax = np.array([r["gt_axis"] for r in rows])
pr_ax = np.array([r["pred_axis"] for r in rows])
dW = np.array([r["d_world"] for r in rows])
npts = np.array([r["gt_npts"] for r in rows])

vert = np.abs(gt_ax[:, 2]) > np.cos(np.deg2rad(20))
horiz = np.abs(gt_ax[:, 2]) < np.sin(np.deg2rad(20))

print("max gt_npts among matched:", npts.max(), " n>8000:", (npts > 8000).sum())

# --- 1. is the -0.23 m z offset real, or an artefact of vertical hinges (z is along-axis)? ---
sec = {}
for tag, m in [("rotation_vertical_axis", rot & vert), ("rotation_horizontal_axis", rot & horiz),
               ("translation_all", ~rot)]:
    if m.sum() == 0:
        continue
    d = dW[m]
    # component of the FULL offset along z
    # perpendicular offset (what the metric sees), decomposed into z and in-plane parts
    perp = []
    for i in np.where(m)[0]:
        e = unit(gt_ax[i])
        dd = dW[i]
        perp.append(dd - np.dot(dd, e) * e)
    perp = np.array(perp)
    sec[tag] = {
        "n": int(m.sum()),
        "raw_offset_mean_xyz": d.mean(0).round(4).tolist(),
        "raw_offset_median_xyz": np.median(d, 0).round(4).tolist(),
        "perp_offset_mean_xyz": perp.mean(0).round(4).tolist(),
        "perp_offset_median_xyz": np.median(perp, 0).round(4).tolist(),
        "perp_offset_median_z": float(np.median(perp[:, 2])),
        "perp_mag_median": float(np.median(np.linalg.norm(perp, axis=1))),
        "frac_perp_z_negative": float((perp[:, 2] < 0).mean()),
        "median_|perp_z| / median_perp_mag": float(np.median(np.abs(perp[:, 2])) / max(np.median(np.linalg.norm(perp, axis=1)), 1e-9)),
    }
    print(f"\n[{tag}] n={m.sum()}")
    print("  raw   offset median xyz:", sec[tag]["raw_offset_median_xyz"])
    print("  perp  offset median xyz:", sec[tag]["perp_offset_median_xyz"],
          " |perp| med=%.3f" % sec[tag]["perp_mag_median"],
          " frac perp_z<0: %.2f" % sec[tag]["frac_perp_z_negative"])
out["origin_bias_by_axis_orientation"] = sec

# --- how much would a global z-correction help horizontal-axis rotations? ---
hm = rot & horiz
if hm.sum():
    best = None
    for dz in np.arange(-0.5, 0.51, 0.01):
        vals = []
        for i in np.where(hm)[0]:
            po = np.array(rows[i]["pred_origin"]) + np.array([0, 0, dz])
            vals.append(perp_dist(po, np.array(rows[i]["gt_origin"]), np.array(rows[i]["gt_axis"])))
        vals = np.array(vals)
        rec = (dz, float((vals < 0.25).mean()), float(np.median(vals)))
        if best is None or rec[1] > best[1]:
            best = rec
    out["oracle_global_z_shift_horizontal_rot"] = {"dz": float(best[0]), "pass0.25": best[1], "median": best[2],
                                                   "baseline_pass0.25": float((O[hm] < 0.25).mean()),
                                                   "baseline_median": float(np.median(O[hm]))}
    print("\noracle global z shift for horizontal-axis rotations:", out["oracle_global_z_shift_horizontal_rot"])

# global 3D shift oracle over all rotations
bestv = None
for dx in np.arange(-0.3, 0.31, 0.05):
    for dy in np.arange(-0.3, 0.31, 0.05):
        for dz in np.arange(-0.4, 0.41, 0.05):
            sh = np.array([dx, dy, dz])
            vals = np.array([perp_dist(np.array(rows[i]["pred_origin"]) + sh,
                                       np.array(rows[i]["gt_origin"]), np.array(rows[i]["gt_axis"]))
                             for i in np.where(rot)[0]])
            p = float((vals < 0.25).mean())
            if bestv is None or p > bestv[1]:
                bestv = (sh.round(3).tolist(), p, float(np.median(vals)))
out["oracle_global_xyz_shift_all_rot"] = {"shift": bestv[0], "pass0.25": bestv[1], "median": bestv[2],
                                          "baseline_pass0.25": float((O[rot] < 0.25).mean())}
print("oracle global xyz shift (all rotations):", out["oracle_global_xyz_shift_all_rot"])

# --- 2. near-orthogonal axis cases ---
near90 = [r for r in rows if r["axis_deg"] > 60]
out["near_orthogonal_cases"] = [
    {k: r[k] for k in ["scene", "gid", "label", "axis_deg", "gt_axis", "pred_axis", "gt_npts", "iou", "score", "origin_perp"]}
    for r in sorted(near90, key=lambda x: -x["axis_deg"])]
print("\n#cases axis_err>60deg:", len(near90))
for r in out["near_orthogonal_cases"]:
    print("  %s gid=%d %s err=%.1f gt=%s pred=%s npts=%d iou=%.2f score=%.2f" % (
        r["scene"], r["gid"], r["label"][:4], r["axis_deg"],
        np.round(r["gt_axis"], 2), np.round(r["pred_axis"], 2), r["gt_npts"], r["iou"], r["score"]))

# --- 3. axis / origin failure overlap (rotation only) ---
ri = np.where(rot)[0]
af = A[ri] >= 15
of = O[ri] >= 0.25
out["failure_overlap_rotation"] = {
    "n": int(len(ri)),
    "both_pass": int((~af & ~of).sum()), "axis_only_fail": int((af & ~of).sum()),
    "origin_only_fail": int((~af & of).sum()), "both_fail": int((af & of).sum()),
    "phi_corr": float(np.corrcoef(af.astype(float), of.astype(float))[0, 1]),
}
print("\nfailure overlap (rotation):", out["failure_overlap_rotation"])

# --- 4. trivial baselines on the matched set ---
bl = {}
zup = np.array([0, 0, 1.0])
bl["axis=z_up_pass15"] = float(np.mean([axis_angle_deg(zup, r["gt_axis"]) < 15 for r in rows]))
bl["axis=z_up_pass15_rot"] = float(np.mean([axis_angle_deg(zup, r["gt_axis"]) < 15 for r in rows if r["label"] == "rotation"]))
bl["axis=z_up_pass15_trans"] = float(np.mean([axis_angle_deg(zup, r["gt_axis"]) < 15 for r in rows if r["label"] == "translation"]))
# per-part PCA longest extent direction as axis baseline
bl["origin=part_centroid_pass0.25_rot"] = float(np.mean(
    [perp_dist(np.array(r["centroid"]), np.array(r["gt_origin"]), np.array(r["gt_axis"])) < 0.25
     for r in rows if r["label"] == "rotation"]))
bl["model_axis_pass15"] = float((A < 15).mean())
bl["model_origin_pass0.25_rot"] = float((O[rot] < 0.25).mean())
out["baselines"] = bl
print("\nbaselines:", json.dumps(bl, indent=1))

# --- 5. per-class AP (exact benchmark) ---
matches, gt_arti, preds = build()


def perclass(crit):
    a = esi.compute_averages(esi.evaluate_matches(matches, crit))
    return {c: {"ap50": float(a["classes"][c]["ap50%"]), "ap": float(a["classes"][c]["ap"])} for c in esi.CLASS_LABELS} | \
           {"all_ap50": float(a["all_ap_50%"])}


pc = {
    "M": perclass(esi.match_criteria_M_pred),
    "MA(15)": perclass(esi.match_criteria_MA_pred),
    "MO(0.25)": perclass(esi.match_criteria_MO_pred),
    "MAO_ST(15,0.25)": perclass(esi.match_criteria_MAO_pred_standard),
    "MA(20)": perclass(make_MA(20)),
    "MA(30)": perclass(make_MA(30)),
    "MO(0.3)": perclass(make_MO(0.30)),
    "MO(0.5)": perclass(make_MO(0.50)),
}
out["per_class_ap"] = pc
print("\nper-class AP50:")
for k, v in pc.items():
    print("  %-18s all=%.4f  rotation=%.4f  translation=%.4f" % (k, v["all_ap50"], v["rotation"]["ap50"], v["translation"]["ap50"]))

# --- 6. AP gap decomposition: oracle-fix the tail ---
# what if every axis error >30deg were fixed (i.e. treat as pass) but <=30 kept at 15deg rule?
def make_MA_tailfix(axis_th, tail_th):
    def f(inst_ref, inst_check, iou_th, gt_first=False):
        ov = float(inst_ref["intersection"]) / (
            inst_check["vert_count"] + inst_ref["vert_count"] - inst_ref["intersection"])
        if not ov > iou_th:
            return False
        d = axis_angle_deg(inst_ref["axis"], inst_check["axis"])
        return (d < axis_th) or (d >= tail_th)
    return f


out["ap_gap_decomposition"] = {
    "AP50_M": float(esi.compute_averages(esi.evaluate_matches(matches, esi.match_criteria_M_pred))["all_ap_50%"]),
    "AP50_axis_15": float(esi.compute_averages(esi.evaluate_matches(matches, esi.match_criteria_MA_pred))["all_ap_50%"]),
    "AP50_axis_15_but_gross>30_fixed": float(esi.compute_averages(esi.evaluate_matches(matches, make_MA_tailfix(15, 30)))["all_ap_50%"]),
    "AP50_axis_15_but_gross>45_fixed": float(esi.compute_averages(esi.evaluate_matches(matches, make_MA_tailfix(15, 45)))["all_ap_50%"]),
}
print("\nAP gap decomposition:", json.dumps(out["ap_gap_decomposition"], indent=1))

json.dump(out, open("/data/challenge/analysis/arti_bias_followup.json", "w"), indent=1)
print("\nwrote /data/challenge/analysis/arti_bias_followup.json")
