#!/bin/bash
# Auto-resume supervisor for v9_fullaug: the run OOMs on rare large-crop
# spikes; main_*.py auto-resumes from last-epoch.ckpt, so just relaunch.
cd /data/challenge/USDNet
LOG=results/v9_fullaug.log
tries=0
while :; do
  if tr '\r' '\n' < "$LOG" | grep -q "Trainer.fit. stopped"; then
    echo "training complete"; break
  fi
  if pgrep -f "python main_instance_segmentation_articulation.*v9_fullaug" >/dev/null; then
    sleep 60; continue
  fi
  tries=$((tries+1))
  if [ $tries -gt 8 ]; then echo "too many restarts, giving up"; break; fi
  echo "restart #$tries $(date '+%H:%M')"
  VAL_EVERY=9999 SAVE_EVERY=5 setsid nohup bash scripts/train_full_aug.sh >> "$LOG" 2>&1 &
  sleep 120
done
