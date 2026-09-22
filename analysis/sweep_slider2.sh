#!/bin/bash
# The first sweep showed the new planar branch LOSING to the old agreement gate
# except at the tightest planarity. Two changes were bundled together:
#   (a) on planar parts, take the PCA normal unconditionally (drop the model
#       agreement gate);
#   (b) on planar parts with a sliding-leaf neighbour, take the in-plane
#       direction instead.
# Setting the leaf-gap window to an unreachable range disables (b) alone, so
# these runs attribute the effect to one change or the other.
cd /data/challenge/USDNet || exit 1
export PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis
PY=$HOME/.conda/envs/usdnet/bin/python
IN=results/ev_v4init/debug/val_preds/preds.pkl
PTS=data/processed/articulate3d_challenge_mov/validation
HAN=results/inference_inter_seg_val/debug/val_preds/preds.pkl

run() {
  PLANAR_TH=$1 DN_LO=$2 DN_HI=$3 $PY /data/challenge/analysis/postprocess2.py \
    "$IN" /tmp/sw2.pkl "$PTS" --steps pca,endpoint --handles "$HAN" >/dev/null 2>&1
  S=$($PY /data/challenge/analysis/score_pkl.py /tmp/sw2.pkl 2>/dev/null \
      | grep -E "AP50_axis |AP50_axis_origin" | awk '{printf "%s ", $2}')
  echo "  planar=$1 dn=$2-$3   axis/joint: $S"
}

echo "old rule (baseline):"
run 0.0 0.004 0.06

echo
echo "(a) unconditional normal only, in-plane flip DISABLED:"
for p in 0.03 0.04 0.06 0.08 0.10 0.15; do run $p 9.0 9.001; done

echo
echo "(a)+(b) with in-plane flip enabled:"
for p in 0.03 0.04 0.06 0.08 0.10 0.15; do run $p 0.004 0.06; done
