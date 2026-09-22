#!/bin/bash
# 03:30 decision point: pause v10 training cleanly, eval the freshest weights on
# val with gold inference flags, print mask AP50 + raw joint for the go/no-go call.
cd /data/challenge/USDNet
while [ "$(date '+%H%M')" -lt "0330" ] || [ "$(date '+%H')" -gt "12" ]; do sleep 120; done
echo "pausing v10 at $(date '+%H:%M')"
echo "Trainer.fit stopped (manual pause for night eval)" >> results/v10_maskfix.log
sleep 5
pkill -f "python main_instance_segmentation_articulation.py general.experiment_name=v10_maskfix" 2>/dev/null
sleep 30
for i in 1 2 3 4 5 6; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 2000 ] && break
  sleep 20
done
cp results/v10_maskfix/last-epoch.ckpt results/v10_maskfix/night_eval.ckpt
LASTEP=$(tr '\r' '\n' < results/v10_maskfix.log | grep -oE "^Epoch [0-9]+: 100%" | tail -1)
echo "evaluating checkpoint at: $LASTEP"
rm -rf results/v10ev_night results/v10ev_night.log
CKPT=./results/v10_maskfix/night_eval.ckpt SAVE_DIR=./results/v10ev_night \
  ORIGIN_AGG=attn AXIS_BIN=true AXIS_RES_SCALE=0.3 \
  bash scripts/infer_mov_val_sweep.sh > results/v10ev_night.log 2>&1
M=$(tr '\r' '\n' < results/v10ev_night.log | grep -A6 "^what_  " | grep average | awk '{print $4}')
J=$(tr '\r' '\n' < results/v10ev_night.log | grep -A6 "^what_MAO_ST" | grep average | awk '{print $4}')
echo "V10 NIGHT EVAL: mask_AP50=$M raw_joint=$J (gold baseline: 0.572 / 0.374)"
