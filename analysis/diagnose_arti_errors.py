"""Diagnose USDNet articulation (axis/origin) error modes on Articulate3D val.

Uses the exact benchmark matching structures (validated to reproduce
AP50=0.5711 / AP50_axis=0.4419 / AP50_origin=0.4502 / AP50_axis_origin=0.3214).
"""
import json
import os
import pickle
import numpy as np

import benchmark.evaluate_semantic_instance as esi
from build_matches import build, GT_TXT_DIR

NPY_DIR = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"
OUT_JSON = "/data/challenge/analysis/arti_error_diagnosis.json"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def axis_angle_deg(a, b):
    c = np.abs(np.dot(unit(a), unit(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def perp_dist(pred_origin, gt_origin, axis):
    d = np.asarray(pred_origin, np.float64) - np.asarray(gt_origin, np.float64)
    ax = np.asarray(axis, np.float64)
    d = d - np.dot(d, ax) * ax / np.linalg.norm(ax)  # benchmark's exact formula
    return float(np.linalg.norm(d))


def q(x, name):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }


# ---------------- greedy matching ----------------
def greedy_match(matches, gt_arti):
    """Per scene, per class: preds sorted by confidence desc, take first unused GT with IoU>0.5."""
    rows = []
    n_gt_total, n_pred_total = 0, 0
    for key, m in matches.items():
        scene = m["scene"]
        xyz = np.load(f"{NPY_DIR}/{scene}.npy")[:, :3].astype(np.float64)
        gt_ids = np.loadtxt(os.path.join(GT_TXT_DIR, scene + ".txt"), dtype=np.int64)
        for label in esi.CLASS_LABELS:
            gts = m["gt"][label]
            preds = sorted(m["pred"][label], key=lambda p: -float(p["confidence"]))
            n_gt_total += len(gts)
            n_pred_total += len(preds)
            used = set()
            for p in preds:
                best = None
                for mg in p["matched_gt"]:
                    iou = float(mg["intersection"]) / (
                        p["vert_count"] + mg["vert_count"] - mg["intersection"]
                    )
                    if iou > 0.5 and mg["instance_id"] not in used:
                        best = (iou, mg)
                        break  # IoU>0.5 match is unique
                if best is None:
                    continue
                iou, g = best
                used.add(g["instance_id"])
                gid = int(g["instance_id"])
                gt_axis = np.asarray(gt_arti[scene][gid]["axis"], np.float64)
                gt_origin = np.asarray(gt_arti[scene][gid]["origin"], np.float64)
                pa = np.asarray(p["axis"], np.float64)
                po = np.asarray(p["origin"], np.float64)
                part = xyz[gt_ids == gid]
                centroid = part.mean(0)
                ext = part.max(0) - part.min(0)
                diag = float(np.linalg.norm(ext))
                # origin error decomposition in GT-axis frame
                d = po - gt_origin
                e = unit(gt_axis)
                along = float(np.dot(d, e))
                dperp = d - along * e
                r = float(np.linalg.norm(dperp))
                # is pred origin pulled toward the part centroid?
                c = centroid - gt_origin
                cperp = c - np.dot(c, e) * e
                cperp_n = float(np.linalg.norm(cperp))
                toward_centroid = float(np.dot(dperp, unit(cperp))) if cperp_n > 1e-9 else 0.0
                rows.append(dict(
                    scene=scene, gid=gid, label=label,
                    sem_id=int(gt_arti[scene][gid]["sem_id"]),
                    iou=float(iou), score=float(p["confidence"]),
                    gt_npts=int(g["vert_count"]), pred_npts=int(p["vert_count"]),
                    axis_deg=axis_angle_deg(pa, gt_axis),
                    origin_perp=perp_dist(po, gt_origin, gt_axis),
                    origin_perp_predaxis=perp_dist(po, gt_origin, pa),
                    origin_l2=float(np.linalg.norm(d)),
                    origin_along=along, origin_toward_centroid=toward_centroid,
                    gt_origin_to_centroid_perp=cperp_n,
                    part_diag=diag, part_ext=[float(v) for v in ext],
                    gt_axis=[float(v) for v in unit(gt_axis)],
                    pred_axis=[float(v) for v in unit(pa)],
                    d_world=[float(v) for v in d],
                    gt_origin=[float(v) for v in gt_origin],
                    pred_origin=[float(v) for v in po],
                    centroid=[float(v) for v in centroid],
                ))
    return rows, n_gt_total, n_pred_total


# ---------------- relaxed-threshold criteria ----------------
def make_MA(axis_th):
    def f(inst_ref, inst_check, iou_th, gt_first=False):
        ov = float(inst_ref["intersection"]) / (
            inst_check["vert_count"] + inst_ref["vert_count"] - inst_ref["intersection"])
        if not ov > iou_th:
            return False
        return axis_angle_deg(inst_ref["axis"], inst_check["axis"]) < axis_th
    return f


def make_MO(origin_th):
    def f(inst_ref, inst_check, iou_th, gt_first=False):
        ov = float(inst_ref["intersection"]) / (
            inst_check["vert_count"] + inst_ref["vert_count"] - inst_ref["intersection"])
        if not ov > iou_th:
            return False
        if esi.ID_TO_LABEL[inst_ref["label_id"]] == "translation":
            return True
        if gt_first:
            po, go, ga = inst_check["origin"], inst_ref["origin"], inst_ref["axis"]
        else:
            po, go, ga = inst_ref["origin"], inst_check["origin"], inst_check["axis"]
        return perp_dist(po, go, ga) < origin_th
    return f


def make_MAO_ST(axis_th, origin_th):
    ma = make_MA(axis_th)

    def f(inst_ref, inst_check, iou_th, gt_first=False):
        if not ma(inst_ref, inst_check, iou_th, gt_first):
            return False
        if esi.ID_TO_LABEL[inst_ref["label_id"]] == "translation":
            return True
        if gt_first:
            po, pa, go, ga = inst_check["origin"], inst_check["axis"], inst_ref["origin"], inst_ref["axis"]
        else:
            po, pa, ga, go = inst_ref["origin"], inst_ref["axis"], inst_check["axis"], inst_check["origin"]
        return perp_dist(po, go, ga) < origin_th and perp_dist(po, go, pa) < origin_th
    return f


def ap50(matches, crit):
    return float(esi.compute_averages(esi.evaluate_matches(matches, crit))["all_ap_50%"])


def main():
    matches, gt_arti, preds = build()
    res = {}

    # sanity: official numbers
    res["official"] = {
        "AP50": ap50(matches, esi.match_criteria_M_pred),
        "AP50_axis": ap50(matches, esi.match_criteria_MA_pred),
        "AP50_origin": ap50(matches, esi.match_criteria_MO_pred),
        "AP50_axis_origin": ap50(matches, esi.match_criteria_MAO_pred_standard),
    }
    print("official reproduction:", res["official"])

    rows, n_gt, n_pred = greedy_match(matches, gt_arti)
    res["coverage"] = {
        "scenes_in_preds": len(matches),
        "gt_instances_total": n_gt,
        "pred_instances_total": n_pred,
        "matched_pairs_iou>0.5": len(rows),
        "recall_iou50": len(rows) / n_gt,
    }
    print("coverage:", res["coverage"])

    A = np.array([r["axis_deg"] for r in rows])
    O = np.array([r["origin_perp"] for r in rows])
    lab = np.array([r["label"] for r in rows])
    npts = np.array([r["gt_npts"] for r in rows])

    def block(mask, tag):
        a, o = A[mask], O[mask]
        d = {
            "n": int(mask.sum()),
            "axis_deg": q(a, "axis"),
            "axis_pass15": float((a < 15).mean()) if mask.sum() else None,
            "axis_pass20": float((a < 20).mean()) if mask.sum() else None,
            "axis_pass30": float((a < 30).mean()) if mask.sum() else None,
            "axis_pass5": float((a < 5).mean()) if mask.sum() else None,
            "axis_pass10": float((a < 10).mean()) if mask.sum() else None,
            "origin_perp_m": q(o, "origin"),
            "origin_pass0.25": float((o < 0.25).mean()) if mask.sum() else None,
            "origin_pass0.3": float((o < 0.30).mean()) if mask.sum() else None,
            "origin_pass0.5": float((o < 0.50).mean()) if mask.sum() else None,
            "origin_pass0.1": float((o < 0.10).mean()) if mask.sum() else None,
            "joint_pass": float(((a < 15) & (o < 0.25)).mean()) if mask.sum() else None,
        }
        print(f"\n--- {tag} (n={d['n']}) ---")
        print("  axis deg   med=%.2f mean=%.2f p75=%.2f p90=%.2f p95=%.2f max=%.1f | <5:%.3f <10:%.3f <15:%.3f <20:%.3f <30:%.3f"
              % (d["axis_deg"]["median"], d["axis_deg"]["mean"], d["axis_deg"]["p75"], d["axis_deg"]["p90"],
                 d["axis_deg"]["p95"], d["axis_deg"]["max"], d["axis_pass5"], d["axis_pass10"],
                 d["axis_pass15"], d["axis_pass20"], d["axis_pass30"]))
        print("  origin m   med=%.3f mean=%.3f p75=%.3f p90=%.3f p95=%.3f max=%.2f | <0.1:%.3f <0.25:%.3f <0.3:%.3f <0.5:%.3f"
              % (d["origin_perp_m"]["median"], d["origin_perp_m"]["mean"], d["origin_perp_m"]["p75"],
                 d["origin_perp_m"]["p90"], d["origin_perp_m"]["p95"], d["origin_perp_m"]["max"],
                 d["origin_pass0.1"], d["origin_pass0.25"], d["origin_pass0.3"], d["origin_pass0.5"]))
        return d

    res["all"] = block(np.ones(len(rows), bool), "ALL matched")
    res["rotation"] = block(lab == "rotation", "rotation")
    res["translation"] = block(lab == "translation", "translation")

    # ---- error concentration: how much of the failure is long tail? ----
    conc = {}
    for tag, arr, th in [("axis", A, 15.0), ("origin_rot", O[lab == "rotation"], 0.25)]:
        fail = arr[arr >= th]
        conc[tag] = {
            "n": int(len(arr)), "n_fail": int(len(fail)),
            "fail_rate": float(len(fail) / len(arr)),
            "fail_err_median": float(np.median(fail)) if len(fail) else None,
            "fail_err_p25": float(np.percentile(fail, 25)) if len(fail) else None,
            "fail_err_p75": float(np.percentile(fail, 75)) if len(fail) else None,
            "near_miss_frac_of_fails_1.33x": float((fail < th * 1.333).mean()) if len(fail) else None,
            "near_miss_frac_of_fails_2x": float((fail < th * 2).mean()) if len(fail) else None,
            "gross_frac_of_fails_>3x": float((fail > th * 3).mean()) if len(fail) else None,
        }
    conc["axis"]["fail_hist_deg"] = np.histogram(A[A >= 15], bins=[15, 20, 30, 45, 60, 75, 90, 180])[0].tolist()
    conc["axis"]["hist_all_deg_bins"] = [0, 2, 5, 10, 15, 20, 30, 45, 60, 75, 90, 180]
    conc["axis"]["hist_all_deg"] = np.histogram(A, bins=[0, 2, 5, 10, 15, 20, 30, 45, 60, 75, 90, 180])[0].tolist()
    Orot = O[lab == "rotation"]
    conc["origin_rot"]["hist_bins_m"] = [0, 0.05, 0.1, 0.15, 0.25, 0.3, 0.5, 0.75, 1.0, 2.0, 100]
    conc["origin_rot"]["hist"] = np.histogram(Orot, bins=[0, 0.05, 0.1, 0.15, 0.25, 0.3, 0.5, 0.75, 1.0, 2.0, 1e9])[0].tolist()
    res["concentration"] = conc
    print("\nconcentration:", json.dumps(conc, indent=1))

    # ---- systematic axis bias: near-90 (orthogonal confusion)? ----
    sysb = {}
    sysb["axis_frac_in_[75,105]"] = float(((A >= 75) & (A <= 90)).mean())  # angle capped at 90 by abs()
    sysb["axis_frac_>60"] = float((A > 60).mean())
    sysb["axis_frac_45_90"] = float(((A > 45) & (A <= 90)).mean())
    # GT axis orientation: vertical (z) vs horizontal
    gt_ax = np.array([r["gt_axis"] for r in rows])
    pr_ax = np.array([r["pred_axis"] for r in rows])
    gt_vert = np.abs(gt_ax[:, 2]) > np.cos(np.deg2rad(20))
    pr_vert = np.abs(pr_ax[:, 2]) > np.cos(np.deg2rad(20))
    gt_horiz = np.abs(gt_ax[:, 2]) < np.sin(np.deg2rad(20))
    sysb["gt_axis_vertical_frac"] = float(gt_vert.mean())
    sysb["gt_axis_horizontal_frac"] = float(gt_horiz.mean())
    sysb["pred_axis_vertical_frac"] = float(pr_vert.mean())
    for tag, mask in [("gt_vertical", gt_vert), ("gt_horizontal", gt_horiz)]:
        if mask.sum():
            sysb[f"axis_err_{tag}"] = {"n": int(mask.sum()), "median": float(np.median(A[mask])),
                                       "pass15": float((A[mask] < 15).mean())}
    # vertical<->horizontal flips
    sysb["gt_vert_pred_horiz"] = int((gt_vert & ~pr_vert & (A > 15)).sum())
    sysb["gt_horiz_pred_vert"] = int((gt_horiz & pr_vert & (A > 15)).sum())
    # does the pred axis match ANOTHER gt instance's axis in the same scene better?
    better_other = 0
    n_fail_axis = 0
    for i, r in enumerate(rows):
        if r["axis_deg"] < 15:
            continue
        n_fail_axis += 1
        best = 180.0
        for gid, g in gt_arti[r["scene"]].items():
            if gid == r["gid"]:
                continue
            best = min(best, axis_angle_deg(np.array(r["pred_axis"]), g["axis"]))
        if best < 15:
            better_other += 1
    sysb["axis_fail_matching_other_gt_axis_in_scene"] = {"n_fail": n_fail_axis, "n_match_other": better_other}
    res["systematic_axis"] = sysb
    print("\nsystematic axis:", json.dumps(sysb, indent=1))

    # ---- systematic origin bias ----
    rot = lab == "rotation"
    dW = np.array([r["d_world"] for r in rows])[rot]
    syso = {
        "mean_world_offset_xyz": dW.mean(0).tolist(),
        "median_world_offset_xyz": np.median(dW, 0).tolist(),
        "mean_abs_world_offset_xyz": np.abs(dW).mean(0).tolist(),
        "mean_along_axis_component": float(np.mean([r["origin_along"] for r in rows if r["label"] == "rotation"])),
        "mean_toward_centroid_component": float(np.mean([r["origin_toward_centroid"] for r in rows if r["label"] == "rotation"])),
        "frac_pulled_toward_centroid": float(np.mean([r["origin_toward_centroid"] > 0 for r in rows if r["label"] == "rotation"])),
        "median_toward_centroid_component": float(np.median([r["origin_toward_centroid"] for r in rows if r["label"] == "rotation"])),
        "median_gt_origin_to_centroid_perp": float(np.median([r["gt_origin_to_centroid_perp"] for r in rows if r["label"] == "rotation"])),
    }
    # compare: perp dist of pred origin vs perp dist of part centroid (a "predict-the-centroid" baseline)
    cen_perp = np.array([perp_dist(np.array(r["centroid"]), np.array(r["gt_origin"]), np.array(r["gt_axis"]))
                         for r in rows if r["label"] == "rotation"])
    syso["centroid_baseline_perp"] = q(cen_perp, "cen")
    syso["centroid_baseline_pass0.25"] = float((cen_perp < 0.25).mean())
    syso["pred_better_than_centroid_frac"] = float((Orot < cen_perp).mean())
    # normalized by part size
    diag = np.array([r["part_diag"] for r in rows])[rot]
    syso["origin_err_over_part_diag"] = q(Orot / np.maximum(diag, 1e-6), "rel")
    res["systematic_origin"] = syso
    print("\nsystematic origin:", json.dumps(syso, indent=1))

    # ---- size buckets ----
    edges = [0, 500, 2000, 8000, 10**9]
    names = ["<500", "500-2k", "2k-8k", ">8k"]
    buckets = {}
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = (npts >= lo) & (npts < hi)
        if m.sum() == 0:
            continue
        mr = m & rot
        buckets[nm] = {
            "n": int(m.sum()),
            "median_gt_npts": float(np.median(npts[m])),
            "axis_median": float(np.median(A[m])), "axis_pass15": float((A[m] < 15).mean()),
            "origin_median_rot": float(np.median(O[mr])) if mr.sum() else None,
            "origin_pass0.25_rot": float((O[mr] < 0.25).mean()) if mr.sum() else None,
            "n_rot": int(mr.sum()),
            "median_part_diag": float(np.median([r["part_diag"] for i, r in enumerate(rows) if m[i]])),
        }
    res["size_buckets"] = buckets
    print("\nsize buckets:", json.dumps(buckets, indent=1))

    # correlation
    res["correlations"] = {
        "spearman_axis_vs_npts": float(_spearman(npts, A)),
        "spearman_origin_vs_npts_rot": float(_spearman(npts[rot], O[rot])),
        "spearman_origin_vs_partdiag_rot": float(_spearman(diag, Orot)),
        "spearman_axis_vs_iou": float(_spearman(np.array([r["iou"] for r in rows]), A)),
        "spearman_axis_vs_score": float(_spearman(np.array([r["score"] for r in rows]), A)),
    }
    print("\ncorrelations:", res["correlations"])

    # ---- per-scene concentration ----
    scenes = sorted(set(r["scene"] for r in rows))
    per_scene = {}
    for s in scenes:
        m = np.array([r["scene"] == s for r in rows])
        mr = m & rot
        per_scene[s] = {
            "n_matched": int(m.sum()),
            "axis_median": float(np.median(A[m])), "axis_pass15": float((A[m] < 15).mean()),
            "origin_pass0.25_rot": float((O[mr] < 0.25).mean()) if mr.sum() else None,
            "n_rot": int(mr.sum()),
        }
    res["per_scene"] = per_scene
    fails_by_scene = sorted(per_scene.items(), key=lambda kv: kv[1]["axis_pass15"])
    print("\nworst scenes by axis_pass15:")
    for s, v in fails_by_scene[:10]:
        print("  %s n=%d axis_pass15=%.2f axis_med=%.1f origin_pass=%s" %
              (s, v["n_matched"], v["axis_pass15"], v["axis_median"], v["origin_pass0.25_rot"]))
    # what fraction of all axis failures live in the worst 5 scenes?
    fail_counts = {s: int(((np.array([r["scene"] for r in rows]) == s) & (A >= 15)).sum()) for s in scenes}
    tot_fail = sum(fail_counts.values())
    top5 = sorted(fail_counts.values(), reverse=True)[:5]
    res["axis_fail_scene_concentration"] = {
        "total_axis_fails": tot_fail,
        "top5_scene_share": float(sum(top5) / tot_fail) if tot_fail else None,
        "n_scenes_with_any_fail": int(sum(1 for v in fail_counts.values() if v > 0)),
        "fails_per_scene": fail_counts,
    }
    print("\naxis fail scene concentration:", {k: v for k, v in res["axis_fail_scene_concentration"].items() if k != "fails_per_scene"})

    # ---- threshold sweeps (exact AP recomputation) ----
    sweep = {"axis": {}, "origin": {}, "joint": {}}
    for th in [15, 20, 25, 30, 45, 90.1]:
        sweep["axis"][str(th)] = ap50(matches, make_MA(th))
        print("AP50_axis @%.4g deg = %.4f" % (th, sweep["axis"][str(th)]))
    for th in [0.25, 0.30, 0.40, 0.50, 1.0, 1e9]:
        sweep["origin"][str(th)] = ap50(matches, make_MO(th))
        print("AP50_origin @%.4g m = %.4f" % (th, sweep["origin"][str(th)]))
    for ath, oth in [(15, 0.25), (20, 0.25), (30, 0.25), (15, 0.3), (15, 0.5), (20, 0.3), (30, 0.5), (90.1, 1e9)]:
        sweep["joint"][f"{ath}deg_{oth}m"] = ap50(matches, make_MAO_ST(ath, oth))
        print("AP50_axis_origin @%.4g deg %.4g m = %.4f" % (ath, oth, sweep["joint"][f"{ath}deg_{oth}m"]))
    res["threshold_sweep"] = sweep

    res["rows"] = rows
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote", OUT_JSON)


def _spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float("nan")
    return spearmanr(x, y).correlation


if __name__ == "__main__":
    main()
