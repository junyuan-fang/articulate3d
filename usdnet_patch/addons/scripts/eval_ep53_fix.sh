#!/bin/bash
# Re-evaluate v5_res ep53 with the res scale it was TRAINED with (0.1).
# infer_mov_val_v2.sh defaults axis_res_scale=0.0, which feeds the raw MLP
# output into the anchors unbounded and destroys every axis (0.4677 -> 0.027).
cd "$(dirname "$0")/.."
rm -rf results/ev_v5ep53b results/ev_v5ep53b.log
CKPT=./results/v5_res/ep53.ckpt SAVE_DIR=./results/ev_v5ep53b \
  ORIGIN_AGG=attn AXIS_BIN=true AXIS_RES_SCALE=0.1 \
  bash scripts/infer_mov_val_v2.sh > results/ev_v5ep53b.log 2>&1

export PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis
PY=$HOME/.conda/envs/usdnet/bin/python
$PY /data/challenge/analysis/postprocess2.py \
  results/ev_v5ep53b/debug/val_preds/preds.pkl /tmp/ep53b_full.pkl \
  data/processed/articulate3d_challenge_mov/validation \
  --steps pca,endpoint \
  --handles results/inference_inter_seg_val/debug/val_preds/preds.pkl 2>/dev/null | tail -1
echo "=== ep53(scale=0.1) raw ==="
$PY /data/challenge/analysis/score_pkl.py results/ev_v5ep53b/debug/val_preds/preds.pkl 2>/dev/null | grep AP50
echo "=== ep53(scale=0.1) + geo stack ==="
$PY /data/challenge/analysis/score_pkl.py /tmp/ep53b_full.pkl 2>/dev/null | grep AP50
