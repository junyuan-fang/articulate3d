#!/bin/bash
# Last TTA variant: selective union (append only genuinely-new flipped
# proposals at halved score), then geo stack, then score.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis
PY=$HOME/.conda/envs/usdnet/bin/python
D=data/processed/articulate3d_challenge_mov/validation

$PY /data/challenge/analysis/merge_flip_tta.py \
  results/ev_v4init/debug/val_preds/preds.pkl \
  results/ev_flipx/debug/val_preds/preds.pkl \
  /tmp/val_tta_new.pkl \
  --consts data/processed/articulate3d_challenge_mov_flipx/flip_constants.json \
  --addnew 0.3 2>/dev/null

$PY /data/challenge/analysis/postprocess2.py /tmp/val_tta_new.pkl /tmp/val_tta_new_geo.pkl $D \
  --steps pca,endpoint \
  --handles results/inference_inter_seg_val/debug/val_preds/preds.pkl 2>/dev/null | tail -1
echo "=== addnew(0.3) + geo stack ==="
$PY /data/challenge/analysis/score_pkl.py /tmp/val_tta_new_geo.pkl 2>/dev/null | grep AP50
