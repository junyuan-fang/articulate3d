#!/bin/bash
cd /data/challenge/USDNet
LOG=results/inter_v2_aug.log
tries=0
while :; do
  tr '\r' '\n' < "$LOG" | grep -q "Trainer.fit. stopped" && { echo "training complete"; break; }
  if pgrep -f "python main_instance_segmentation_articulation.*inter_v2_aug" >/dev/null; then sleep 45; continue; fi
  tries=$((tries+1)); [ $tries -gt 8 ] && { echo "too many restarts"; break; }
  echo "restart #$tries $(date '+%H:%M')"
  CROPPING=true setsid nohup bash scripts/train_inter_aug.sh >> "$LOG" 2>&1 &
  sleep 120
done
