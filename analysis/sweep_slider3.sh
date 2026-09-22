#!/bin/bash
# Final form: keep the proven agreement gate everywhere, and ONLY add the
# sliding-leaf override on top of it. sweep_slider2.sh showed the two bundled
# changes pull in opposite directions -- dropping the gate costs ~0.010 AP50_axis,
# the leaf flip gains more than that back.
#
# A threshold read off val is worth about a third of its face value on test
# (docs/05-improvement.md), so what matters here is not the peak but whether the
# gain holds across a wide band of settings.
cd /data/challenge/USDNet || exit 1
export PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis
PY=$HOME/.conda/envs/usdnet/bin/python
IN=results/ev_v4init/debug/val_preds/preds.pkl
PTS=data/processed/articulate3d_challenge_mov/validation
HAN=results/inference_inter_seg_val/debug/val_preds/preds.pkl

run() {
  PLANAR_TH=$1 DN_LO=$2 DN_HI=$3 $PY /data/challenge/analysis/postprocess2.py \
    "$IN" /tmp/sw3.pkl "$PTS" --steps pca,endpoint --handles "$HAN" >/dev/null 2>&1
  S=$($PY /data/challenge/analysis/score_pkl.py /tmp/sw3.pkl 2>/dev/null \
      | grep -E "AP50 |AP50_axis |AP50_origin |AP50_axis_origin" | awk '{printf "%s ", $2}')
  echo "  planar=$1 dn=$2-$3   AP50/axis/origin/joint: $S"
}

echo "baseline (leaf override disabled):"
run 0.15 9.0 9.001

echo
echo "planarity band, leaf gap 0.004-0.06:"
for p in 0.03 0.04 0.05 0.06 0.08 0.10 0.15 0.25; do run $p 0.004 0.06; done

echo
echo "leaf-gap band, planarity 0.06:"
for w in "0.002 0.05" "0.004 0.05" "0.004 0.06" "0.004 0.08" "0.006 0.06" "0.008 0.06" "0.004 0.12"; do
  run 0.06 $w
done
