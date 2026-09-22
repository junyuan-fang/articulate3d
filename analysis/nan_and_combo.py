"""(1) Quantify the arccos NaN edge case in the official evaluator when pred axis == gt axis exactly.
   (2) Run the best combo interventions."""
import json
import pickle
import numpy as np

import benchmark.evaluate_semantic_instance as esi
from build_matches import set_articulate3d_labels, PRED_PKL
from interventions import build_from, aps, unit, NPY_DIR

set_articulate3d_labels()
out = {}

# ---- (1) NaN demo, exactly as the benchmark computes it ----
z = np.array([0, 0, 1.0])
gt_examples = []
import h5py, os, glob
n_nan = 0
n_tot = 0
for f in sorted(glob.glob(f"{NPY_DIR}/*_articulation.h5")):
    with h5py.File(f, "r") as h:
        for k in h.keys():
            ax = np.array(h[k]["axis"]).astype(np.float64)
            n_tot += 1
            d = np.dot(z, ax) / (np.linalg.norm(z) * np.linalg.norm(ax))
            d = np.abs(d)
            if np.isnan(np.arccos(d)):
                n_nan += 1
                gt_examples.append((os.path.basename(f), k, ax.tolist(), float(d)))
out["nan_check_snap_to_exact_z"] = {
    "n_gt_axes_total": n_tot, "n_gt_axes_giving_nan_with_pred_z": n_nan,
    "examples": gt_examples[:5],
    "note": "arccos(|cos|) with |cos| slightly > 1 -> nan -> `nan < th` is False -> the official "
            "evaluator marks a PERFECT axis prediction as a FAILURE.",
}
print(json.dumps(out["nan_check_snap_to_exact_z"], indent=1)[:1200])

# ---- (2) combos ----
base = pickle.load(open(PRED_PKL, "rb"))
xyz = {s: np.load(f"{NPY_DIR}/{s}.npy")[:, :3].astype(np.float64) for s in base}


def copy_preds():
    return {s: {k: (v.copy() if hasattr(v, "copy") else v) for k, v in p.items()} for s, p in base.items()}


def gravity_snap(p, th=30, eps=0.0):
    ax = p["pred_axises"].astype(np.float64)
    for i in range(ax.shape[0]):
        if p["pred_classes"][i] == 1:
            a = unit(ax[i])
            if np.abs(a @ z) > np.cos(np.deg2rad(th)):
                v = z * np.sign(a @ z)
                if eps > 0:  # tiny tilt to dodge the arccos NaN edge case
                    v = unit(v + eps * np.array([1.0, 0, 0]))
                ax[i] = v
    p["pred_axises"] = ax.astype(np.float32)


def snap_origin_to_mask(p, s):
    o = p["pred_origins"].astype(np.float64)
    m = base[s]["pred_masks"] > 1e-3
    for i in range(o.shape[0]):
        if p["pred_classes"][i] == 1 and m[:, i].sum() > 0:
            pts = xyz[s][m[:, i]]
            o[i] = pts[np.argmin(((pts - o[i]) ** 2).sum(1))]
    p["pred_origins"] = o.astype(np.float32)


variants = {}
pm = copy_preds()
for s, p in pm.items():
    gravity_snap(p, 30, eps=1e-3)
variants["A30_eps"] = pm

pm = copy_preds()
for s, p in pm.items():
    gravity_snap(p, 30, eps=1e-3)
    snap_origin_to_mask(p, s)
variants["A30_eps+D"] = pm

pm = copy_preds()
for s, p in pm.items():
    gravity_snap(p, 30, eps=0.0)
    snap_origin_to_mask(p, s)
variants["A30_exact+D"] = pm

for k, pm in variants.items():
    out[k] = aps(build_from(pm))
    print(k, json.dumps(out[k]))

json.dump(out, open("/data/challenge/analysis/arti_nan_combo.json", "w"), indent=1)
print("wrote /data/challenge/analysis/arti_nan_combo.json")
