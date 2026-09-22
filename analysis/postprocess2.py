#!/usr/bin/env python
"""Training-free post-processing, v2. Only touches axes and origins, never
pred_masks, so AP50 is unchanged by construction.

Three stacked interventions (docs/05-improvement.md):

1. normalize -- rescale every axis to unit length. This is not cosmetic: the
   leaderboard criterion match_criteria_MAO_pred_standard projects with
       d - dot(d, pred_axis) * pred_axis / norm(pred_axis)
   where a correct projection needs norm**2. GT axes are unit so their branch is
   fine, but the model emits axes with norm 0.11-1.46, so normalizing changes
   the verdict.

2. canonical -- snap an axis to the nearest canonical direction when it is
   already within a threshold. 96.4% of GT axes sit within 5 deg of canonical
   (rotation overwhelmingly vertical, translation horizontal), and the axis head
   has collapsed onto that prior anyway.

3. endpoint -- move each origin to one of the two ends of its own predicted
   mask, measured along the widest direction PERPENDICULAR TO THE AXIS. A door
   hinged about a vertical axis has its hinge on a lateral edge, so the relevant
   extent is the one across the axis, not the part's longest one.

   ** This is a correction. ** The first version of this file used the mask's
   first PCA direction, which for a tall door is vertical -- it snapped origins
   to the top/bottom of the part and cost 0.19 AP50_origin end to end:

       norm+canon                     AP50_origin 0.4495   joint 0.3550
       norm+canon+endpoint(1st PCA)   AP50_origin 0.2619   joint 0.1898  <-- bad
       norm+canon+endpoint(perp)      AP50_origin 0.4631   joint 0.3828

   Uses robust 5%/95% quantiles; raw min/max is outlier-driven and scores worse
   (joint 0.3435). Verified by analysis/pp_sweep.py on the val preds.

Usage:
    python postprocess2.py IN.pkl OUT.pkl POINTS_DIR [--steps normalize,canonical,endpoint]
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np

Z = np.array([0.0, 0.0, 1.0])
ROTATION, TRANSLATION = 1, 2
MASK_THRESH = 1e-3
CANON_DEG = 30.0
# Swept against val (analysis/sweep_geo.py). The curve is unimodal with a broad
# plateau, which is why this is trusted rather than treated as threshold fitting:
#   2/98 .4187 | 5/95 .4306 | 10/90 .4359 | 12/88 .4354 | 15/85 .4345
#   | 20/80 .4292 | 25/75 .3810 | 30/70 .3027
# Past ~20 the "endpoint" is no longer an endpoint, hence the collapse.
QLO, QHI = 10.0, 90.0
# Bbox margin for deciding a predicted handle belongs to this part. Swept in
# analysis/sweep_geo2.py: 0.10/0.15 -> .4395 (plateau), 0.25/0.35 -> .4359,
# 0.50 -> .4330. A wide margin lets a NEIGHBOURING part's handle vote, which
# flips the hinge side. CANON_DEG (15-45) and the handle score cut (0.3-0.7)
# were swept in the same run and are completely flat -- both saturated.
HANDLE_MARGIN = 0.12
# Sliding-leaf detection for translation axes (see pca_translation_axes).
# DN_LO/DN_HI: plane separation of a neighbouring leaf, in metres -- above zero
#   because coplanar fronts cannot pass each other, and around a leaf thickness
#   rather than a room's width. Completely flat on val over 0.004-0.008 x
#   0.05-0.08 (analysis/sweep_slider3.sh); only 0.002 breaks it, by admitting
#   the exactly-coplanar drawer rows.
# PLANAR_TH / HEIGHT_TOL: how flat a panel must be, and how closely the two
#   leaves must match in height. The full grid is in analysis/sweep_slider5.sh;
#   against a no-flip baseline of 0.4395 joint:
#       planar\htol   0.10    0.15    0.20    0.25    0.30
#       0.03        0.4406  0.4432  0.4432  0.4432  0.4432
#       0.05        0.4605  0.4605  0.4605  0.4605  0.4605
#       0.08        0.4530  0.4530  0.4477  0.4477  0.4413
#       0.12        0.4484  0.4484  0.4432  0.4414  0.4351
#       0.20        0.4484  0.4484  0.4432  0.4378  0.4316
#       0.35        0.4460  0.4439  0.4361  0.4296  0.4235
#   At htol <= 0.15 every planarity from 0.03 to 0.35 beats the baseline, so the
#   settings below sit on a plateau rather than a fitted peak. Requiring the
#   matched pair is what bought that: the one-sided version (HEIGHT_TOL large)
#   scored BELOW baseline outside a 0.02-wide window in planarity.
PLANAR_TH = float(os.environ.get("PLANAR_TH", 0.06))
DN_LO = float(os.environ.get("DN_LO", 0.004))
DN_HI = float(os.environ.get("DN_HI", 0.06))
PARALLEL_DEG = 20.0
MUTUAL = os.environ.get("MUTUAL", "1") == "1"
HEIGHT_TOL = float(os.environ.get("HEIGHT_TOL", 0.15))
MIN_LEAF_H = float(os.environ.get("MIN_LEAF_H", 0.45))


def normalize_axes(axes):
    axes = np.asarray(axes, dtype=np.float64)
    n = np.linalg.norm(axes, axis=1, keepdims=True)
    return axes / np.maximum(n, 1e-9)


def snap_canonical(axes, classes, thresh_deg=CANON_DEG):
    """Rotation -> vertical; translation -> horizontal plane. Sign-agnostic."""
    axes = normalize_axes(axes).copy()
    classes = np.asarray(classes).ravel()
    cos_z = np.abs(axes @ Z)
    lim = np.cos(np.deg2rad(thresh_deg))

    rot = (classes == ROTATION) & (cos_z >= lim)
    axes[rot] = Z

    # translation axes are horizontal: within thresh of the xy-plane means the
    # z component is small, so project it out and renormalise
    tr = (classes == TRANSLATION) & (cos_z <= np.cos(np.deg2rad(90 - thresh_deg)))
    if tr.any():
        flat = axes[tr].copy()
        flat[:, 2] = 0.0
        nrm = np.linalg.norm(flat, axis=1, keepdims=True)
        keep = nrm.ravel() > 1e-6
        flat[keep] /= nrm[keep]
        axes[np.where(tr)[0][keep]] = flat[keep]
    return axes


def _panel(pts):
    """Flat-panel description of a part: horizontal normal, the in-plane
    horizontal direction perpendicular to it, and how planar the part is."""
    c = pts - pts.mean(0)
    if len(c) > 4000:
        c = c[np.random.RandomState(0).choice(len(c), 4000, replace=False)]
    _, sv, vt = np.linalg.svd(c, full_matrices=False)
    planarity = float(sv[2] / max(sv[1], 1e-9))
    nrm = None
    for v in (vt[2], vt[1]):
        f = v.copy(); f[2] = 0.0
        n = np.linalg.norm(f)
        if n > 0.3:
            nrm = f / n
            break
    if nrm is None:
        return None
    return dict(nrm=nrm, inp=np.array([-nrm[1], nrm[0], 0.0]), planarity=planarity,
                cen=pts.mean(0), h=float(np.ptp(pts[:, 2])),
                w=float(np.ptp((pts - pts.mean(0)) @ np.array([-nrm[1], nrm[0], 0.0]))))


def _is_sliding_leaf(me, others):
    """True when this panel belongs to a set of side-by-side sliding leaves.

    Two leaves of a sliding door or window sit in DIFFERENT planes, about one
    leaf thickness apart, so that one can pass behind the other, and they stand
    beside each other at the same height. A stack or row of drawers shares one
    plane exactly (offset ~ 0) -- coplanar fronts cannot pass each other -- and
    panels metres apart belong to different furniture.

    Measured on val (analysis/test_neighbour_rule.py), the plane offset splits
    the two classes with no overlap: the nine in-plane sliders sit at
    0.008-0.035 m, and the drawer-like parts at 0.000, 0.089, 0.245, 1.30, 1.84.
    """
    for o in others:
        cos = abs(float(np.dot(me["nrm"], o["nrm"])))
        if cos < np.cos(np.deg2rad(PARALLEL_DEG)):
            continue
        d = o["cen"] - me["cen"]
        dn = abs(float(d @ me["nrm"]))
        di = abs(float(d @ me["inp"]))
        dz = abs(float(d[2]))
        if not (DN_LO < dn < DN_HI):
            continue
        if MUTUAL:
            # leaves come as a matched pair: the partner is a panel too, and
            # the two are cut to the same height. Stating the pair symmetrically
            # is what carries over where a one-sided planarity cut does not.
            if o["planarity"] >= PLANAR_TH:
                continue
            if abs(me["h"] - o["h"]) > HEIGHT_TOL * max(me["h"], o["h"]):
                continue
            # ...and a sliding leaf is door-sized. A row of drawer fronts is
            # also same-height and near-coplanar, so without this the rule
            # fires on them: on test it was flipping 0.23 m and 0.33 m panels,
            # one of them the highest-scoring prediction in its scene. Every
            # true in-plane case on val is at least 0.50 m tall, so this costs
            # nothing there and only removes parts too short to be a door.
            if min(me["h"], o["h"]) < MIN_LEAF_H:
                continue
        if di > 0.2 * (me["w"] + o["w"]) and dz < 0.5 * min(me["h"], o["h"]):
            return True
    return False


def pca_translation_axes(axes, classes, masks, points, agree_deg=25.0):
    """Set a translation axis from the part's own panel geometry.

    A drawer/sliding-door front is a flat panel, so its slide direction is one
    of exactly two horizontal directions: the panel normal (a drawer travels out
    through its own face) or the in-plane horizontal (a sliding leaf travels
    across it). Measured on val (analysis/verify_geo.py, 95 matched
    translations) the PCA panel normal has median azimuth error 0.70 deg against
    the model head's 5.54 deg, so the geometry is far more accurate than the
    network -- the whole problem is choosing between the two directions.

    The model cannot make that choice: it has learnt a drawers-slide-outward
    prior and sits closer to the normal on every one of the nine in-plane cases
    (analysis/diag_inplane.py). So a part is flipped to the in-plane direction
    only on the geometric evidence of a sliding partner; everything else keeps
    the agreement gate, which accepts the panel normal when the model already
    leans that way.

    Val, whole pipeline (analysis/sweep_slider5.sh):

        AP50               0.5716 -> 0.5716    untouched by construction
        AP50_origin        0.5018 -> 0.5018    untouched
        AP50_axis          0.4836 -> 0.5046    +0.0210
        AP50_axis_origin   0.4395 -> 0.4605    +0.0210

    Two things did NOT work and are recorded so they are not retried:
      * taking the panel normal unconditionally on planar parts, instead of
        through the agreement gate, costs ~0.010 AP50_axis. Matched-instance
        retention says it should gain, but AP integrates the whole
        precision-recall curve and panel geometry is unreliable on the
        low-confidence predictions filling its tail. Trust the AP number.
      * letting the model choose between the two directions. It picks the
        normal on 5/5 of the cases that need in-plane.
    """
    axes = np.asarray(axes, dtype=np.float64).copy()
    classes = np.asarray(classes).ravel()
    n_replaced = 0

    panels = {}
    for i in range(axes.shape[0]):
        if classes[i] != TRANSLATION:
            continue
        sel = masks[:, i] > MASK_THRESH
        if sel.sum() < 10:
            continue
        p = _panel(points[sel])
        if p is not None:
            panels[i] = p

    for i, p in panels.items():
        others = [q for j, q in panels.items() if j != i]
        if p["planarity"] < PLANAR_TH and _is_sliding_leaf(p, others):
            axes[i] = p["inp"]
            n_replaced += 1
            continue
        # everything else keeps the agreement gate, which only accepts the PCA
        # normal when the model already leans that way. Taking the normal
        # unconditionally on planar parts was tried and is worse -- val
        # AP50_axis 0.474-0.478 against the gate's 0.4836
        # (analysis/sweep_slider2.sh), because AP integrates the whole
        # precision-recall curve and panel geometry is unreliable on the
        # low-confidence predictions that fill out its tail.
        cur = axes[i] / max(np.linalg.norm(axes[i]), 1e-9)
        ang = np.degrees(np.arccos(min(1.0, abs(float(np.dot(cur, p["nrm"]))))))
        if ang < agree_deg:
            axes[i] = p["nrm"]
            n_replaced += 1
    return axes, n_replaced


def snap_origin_endpoints(origins, axes, classes, masks, points, handles=None):
    """Snap each rotation origin to an endpoint of its own mask.

    The endpoint is taken along the mask's widest direction *perpendicular to the
    predicted axis* -- the lateral edge a door is hinged on -- at the 5%/95%
    quantiles of the projection. Using the part's overall principal direction
    instead is what the previous version did and it loses 0.19 AP50_origin.

    Side selection: "the hinge is on the side opposite the handle". When
    predicted interactable (handle) centroids are supplied and one lies near the
    part, pick the endpoint FARTHER from it; otherwise fall back to the endpoint
    nearest the model's predicted origin. Measured on val matched rotations
    (analysis/verify_handle_side.py): origin<0.25 pass 78.3% -> 82.8%.

    Translation instances are left alone: the metric ignores their origin.
    """
    origins = np.asarray(origins, dtype=np.float64).copy()
    classes = np.asarray(classes).ravel()
    masks = np.asarray(masks)
    n_moved = 0
    for i in range(origins.shape[0]):
        if classes[i] != ROTATION:
            continue
        sel = masks[:, i] > MASK_THRESH
        if sel.sum() < 3:
            continue
        pts = points[sel]
        a = np.asarray(axes[i], dtype=np.float64)
        a = a / max(np.linalg.norm(a), 1e-9)
        c = pts.mean(axis=0)
        X = pts - c
        X = X - np.outer(X @ a, a)              # project onto the plane cross the axis
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        d = vt[0] - (vt[0] @ a) * a
        d = d / max(np.linalg.norm(d), 1e-9)
        t = X @ d
        lo, hi = np.percentile(t, [QLO, QHI])
        cand = np.stack([c + lo * d, c + hi * d])
        k = None
        if handles is not None and len(handles):
            bbox_lo, bbox_hi = pts.min(0) - HANDLE_MARGIN, pts.max(0) + HANDLE_MARGIN
            near = handles[np.all((handles > bbox_lo) & (handles < bbox_hi), axis=1)]
            if len(near):
                dh0 = np.linalg.norm(near - cand[0], axis=1).min()
                dh1 = np.linalg.norm(near - cand[1], axis=1).min()
                k = int(dh1 > dh0)              # hinge opposite the handle
        if k is None:
            # fall back: the endpoint the model already leans toward
            k = int(np.argmin(np.linalg.norm(cand - origins[i], axis=1)))
        origins[i] = cand[k]
        n_moved += 1
    return origins, n_moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("points_dir")
    ap.add_argument("--steps", default="normalize,canonical,endpoint")
    ap.add_argument("--canon-deg", type=float, default=CANON_DEG)
    ap.add_argument("--handles", default=None,
                    help="inter-model preds.pkl; enables handle-opposite side selection")
    args = ap.parse_args()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    handle_cent = {}
    if args.handles:
        inter = pickle.load(open(args.handles, "rb"))
        for scene, ip in inter.items():
            im = np.asarray(ip["pred_masks"]) > MASK_THRESH
            isc = np.asarray(ip["pred_scores"]).ravel()
            cents = []
            for qi in np.where(isc > 0.5)[0]:
                sel = im[:, qi]
                if 10 <= sel.sum() < 20000:
                    cents.append(sel.copy())    # column copy; centroid needs scene pts
            handle_cent[scene] = cents
        del inter

    preds = pickle.load(open(args.inp, "rb"))
    pdir = Path(args.points_dir)
    moved = 0
    for scene, p in preds.items():
        if "normalize" in steps:
            p["pred_axises"] = normalize_axes(p["pred_axises"])
        if "canonical" in steps:
            p["pred_axises"] = snap_canonical(p["pred_axises"], p["pred_classes"],
                                              args.canon_deg)
        if "pca" in steps or "endpoint" in steps:
            pts = np.load(pdir / f"{scene}.npy", mmap_mode="r")[:, 0:3].astype(np.float64)
            if pts.shape[0] != np.asarray(p["pred_masks"]).shape[0]:
                raise SystemExit(f"{scene}: point/mask row mismatch -- wrong points_dir?")
        if "pca" in steps:
            p["pred_axises"], n = pca_translation_axes(
                p["pred_axises"], p["pred_classes"],
                np.asarray(p["pred_masks"]), pts)
            moved += n
        if "endpoint" in steps:
            hs = None
            if scene in handle_cent and handle_cent[scene]:
                hs = np.array([pts[sel].mean(0) for sel in handle_cent[scene]])
            p["pred_origins"], n = snap_origin_endpoints(
                p["pred_origins"], p["pred_axises"], p["pred_classes"],
                p["pred_masks"], pts, handles=hs)
            moved += n

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(preds, open(args.out, "wb"))
    print(f"{len(preds)} scenes | steps={steps} | origins moved: {moved}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
