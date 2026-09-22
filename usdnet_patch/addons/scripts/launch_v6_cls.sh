#!/bin/bash
# Stage B: anchor-classifier-only training, single-layer rebalance.
#
# Target: the 7 lateral hinges the bin prior snaps to gravity (90 deg errors,
# the one axis failure mode geometry cannot fix). One variable: axis_cls_head
# (~29K params). Rebalance via loss weight 2 ONLY -- original database, no hbal
# (v4 showed stacking both pushes the training prior to ~1:1.4 against a 13:1
# deployment world and collapses the head).
#
# Watch: val every 2 epochs. The failure signature to abort on is AP50_axis
# dropping below the 0.4728-level warm start -- that means the classifier is
# flipping vertical hinges to horizontal.
cd "$(dirname "$0")/.."
rm -rf results/v6_cls results/v6_cls.log
EXP=v6_cls TRAIN_HEADS_V2=false TRAIN_CLS_ONLY=true \
  CKPT=./results/v3_origin_bnfix/ep19.ckpt \
  TRAIN_DB=data/processed/articulate3d_challenge_mov \
  AXIS_RES_SCALE=0.1 AXIS_HORIZ_WEIGHT=2.0 NEW_HEAD_LR_MULT=1.0 \
  EPOCHS=60 VAL_EVERY=2 \
  setsid nohup bash scripts/train_arti_v4.sh > results/v6_cls.log 2>&1 &
disown
echo "launched v6_cls"
