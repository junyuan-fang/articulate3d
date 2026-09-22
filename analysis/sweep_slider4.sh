#!/bin/bash
# sweep_slider3.sh found the leaf flip worth +0.0124 joint at planarity 0.05,
# but as a spike: 0.03 and 0.08 both score BELOW the no-flip baseline. A gain
# that inverts within +/-0.02 of its threshold is fitted to the one val scene
# that supplies eight of the nine positives, not a transferable rule.
#
# So state the pair symmetrically instead -- both panels flat, both cut to the
# same height -- and check whether that flattens the planarity curve. A wide
# band that stays above baseline is deployable; another spike is not.
cd /data/challenge/USDNet || exit 1
export PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis
PY=$HOME/.conda/envs/usdnet/bin/python
IN=results/ev_v4init/debug/val_preds/preds.pkl
PTS=data/processed/articulate3d_challenge_mov/validation
HAN=results/inference_inter_seg_val/debug/val_preds/preds.pkl

run() {
  MUTUAL=$4 HEIGHT_TOL=$5 PLANAR_TH=$1 DN_LO=$2 DN_HI=$3 \
    $PY /data/challenge/analysis/postprocess2.py \
    "$IN" /tmp/sw4.pkl "$PTS" --steps pca,endpoint --handles "$HAN" >/dev/null 2>&1
  S=$($PY /data/challenge/analysis/score_pkl.py /tmp/sw4.pkl 2>/dev/null \
      | grep -E "AP50_axis |AP50_axis_origin " | awk '{printf "%s ", $2}')
  echo "  planar=$1 htol=$5   axis/joint: $S"
}

echo "baseline (no leaf flip):            axis/joint: 0.4836 0.4395"
echo
echo "mutual pair, height tol 0.35:"
for p in 0.03 0.04 0.05 0.06 0.08 0.10 0.15 0.25; do run $p 0.004 0.06 1 0.35; done
echo
echo "mutual pair, height tol 0.20:"
for p in 0.04 0.06 0.10 0.15 0.25; do run $p 0.004 0.06 1 0.20; done
echo
echo "mutual pair, height tol 0.50:"
for p in 0.04 0.06 0.10 0.15 0.25; do run $p 0.004 0.06 1 0.50; done
