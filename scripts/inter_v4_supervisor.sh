#!/bin/bash
cd /data/challenge/USDNet
LOG=results/inter_v4_long.log
tries=0
while :; do
  tr '\r' '\n' < "$LOG" | grep -q "Trainer.fit. stopped" && { echo "training complete"; break; }
  alive=$(python3 -c "
import os
me=os.getpid()
n=0
for pid in os.listdir('/proc'):
    if not pid.isdigit(): continue
    try: cmd=open(f'/proc/{pid}/cmdline','rb').read().decode(errors='ignore')
    except Exception: continue
    if 'main_instance_segmentation' in cmd and 'inter_v4' + '_long' in cmd: n+=1
print(n)")
  if [ "$alive" -gt 0 ]; then sleep 60; continue; fi
  tries=$((tries+1)); [ $tries -gt 15 ] && { echo "too many restarts"; break; }
  echo "restart #$tries $(date '+%H:%M')"
  CROPPING=true EXP=inter_v4_long EPOCHS=300 LR=0.00002 setsid nohup bash scripts/train_inter_aug.sh >> "$LOG" 2>&1 &
  sleep 150
done
