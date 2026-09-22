#!/bin/bash
# The matched-pair predicate turned the planarity spike into a plateau: at
# height tolerance 0.20 every planarity from 0.04 to 0.25 beats the no-flip
# baseline of 0.4395, where the one-sided version fell below it outside a
# 0.02-wide window. Map how far that plateau actually extends in both
# parameters before choosing where to sit on it -- the point of the exercise is
# to pick a setting whose neighbours are all safe, not the highest number.
cd /data/challenge/USDNet || exit 1
export PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis
PY=$HOME/.conda/envs/usdnet/bin/python
IN=results/ev_v4init/debug/val_preds/preds.pkl
PTS=data/processed/articulate3d_challenge_mov/validation
HAN=results/inference_inter_seg_val/debug/val_preds/preds.pkl

run() {
  MUTUAL=1 HEIGHT_TOL=$2 PLANAR_TH=$1 DN_LO=0.004 DN_HI=0.06 \
    $PY /data/challenge/analysis/postprocess2.py \
    "$IN" /tmp/sw5.pkl "$PTS" --steps pca,endpoint --handles "$HAN" >/dev/null 2>&1
  $PY /data/challenge/analysis/score_pkl.py /tmp/sw5.pkl 2>/dev/null \
    | awk '/AP50_axis_origin/{printf "%s ", $2}'
}

echo "baseline (no leaf flip): joint 0.4395"
printf "\n%-8s" "planar\\htol"
for h in 0.10 0.15 0.20 0.25 0.30; do printf "%8s" "$h"; done
echo
for p in 0.03 0.05 0.08 0.12 0.20 0.35; do
  printf "%-8s" "$p"
  for h in 0.10 0.15 0.20 0.25 0.30; do printf "%8s" "$(run $p $h)"; done
  echo
done
