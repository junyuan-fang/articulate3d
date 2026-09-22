#!/bin/bash
# Auto-resume supervisor for v10_maskfix: the run OOMs on rare large-crop
# spikes; main_*.py auto-resumes from last-epoch.ckpt, so just relaunch.
cd /data/challenge/USDNet
LOG=results/v10_maskfix.log
tries=0
while :; do
  if tr '\r' '\n' < "$LOG" | grep -q "Trainer.fit. stopped"; then
    echo "training complete"; break
  fi
  if pgrep -f "python main_instance_segmentation_articulation.*v10_maskfix" >/dev/null; then
    sleep 60; continue
  fi
  tries=$((tries+1))
  if [ $tries -gt 20 ]; then echo "too many restarts, giving up"; break; fi
  echo "restart #$tries $(date '+%H:%M')"
  C2F=false EXP=v10_maskfix EPOCHS=300 LR=0.00003 VAL_EVERY=9999 SAVE_EVERY=10 CROP_LEN=3.6 setsid nohup bash scripts/train_full_aug.sh >> "$LOG" 2>&1 &
  sleep 120
done
