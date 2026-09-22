#!/bin/bash
# Stage A launcher: res-only, risk-clamped (docs/06 section 12 aftermath).
cd "$(dirname "$0")/.."
rm -rf results/v5_res results/v5_res.log
EXP=v5_res TRAIN_HEADS_V2=false TRAIN_RES_ONLY=true \
  TRAIN_DB=data/processed/articulate3d_challenge_mov \
  AXIS_RES_SCALE=0.1 AXIS_HORIZ_WEIGHT=1.0 NEW_HEAD_LR_MULT=1.0 \
  EPOCHS=60 VAL_EVERY=3 \
  setsid nohup bash scripts/train_arti_v4.sh > results/v5_res.log 2>&1 &
disown
echo "launched v5_res"
