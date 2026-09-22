#!/bin/bash
# Compare the sliding-leaf panel rule against the old agreement gate, and check
# that the gain sits on a plateau rather than on a fitted edge.
#
# PLANAR_TH=0 disables the planar branch entirely, so every part falls through
# to the old "accept the PCA normal only when the model already agrees" gate --
# that run reproduces the previously deployed pipeline exactly.
cd /data/challenge/USDNet || exit 1
export PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis
PY=$HOME/.conda/envs/usdnet/bin/python
IN=results/ev_v4init/debug/val_preds/preds.pkl
PTS=data/processed/articulate3d_challenge_mov/validation
HAN=results/inference_inter_seg_val/debug/val_preds/preds.pkl

run() {
  PLANAR_TH=$1 DN_LO=$2 DN_HI=$3 $PY /data/challenge/analysis/postprocess2.py \
    "$IN" /tmp/sw.pkl "$PTS" --steps pca,endpoint --handles "$HAN" >/dev/null 2>&1
  S=$($PY /data/challenge/analysis/score_pkl.py /tmp/sw.pkl 2>/dev/null \
      | grep -E "AP50 |AP50_axis |AP50_origin |AP50_axis_origin" \
      | awk '{printf "%s ", $2}')
  echo "planar=$1 dn=$2-$3   AP50/axis/origin/joint: $S"
}

echo "=== old rule (planar branch off) ==="
run 0.0 0.004 0.06
echo
echo "=== planarity threshold ==="
for p in 0.06 0.10 0.15 0.20 0.30; do run $p 0.004 0.06; done
echo
echo "=== leaf-gap window (planarity 0.15) ==="
for lo in 0.002 0.004 0.008; do
  for hi in 0.04 0.06 0.08; do run 0.15 $lo $hi; done
done
