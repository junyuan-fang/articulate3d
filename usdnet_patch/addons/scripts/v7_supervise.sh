#!/bin/bash
# Babysit the v7 trainval run: an eval job running alongside it on the same
# GPU fluctuates between 5 and 13 GB and has already OOM-killed v7 twice. Checkpoints land every epoch, and train_arti_v2.sh
# auto-resumes from last-epoch.ckpt when SAVE_DIR contains one -- so each
# crash costs at most one epoch. Fresh start only when no checkpoint exists.
set -u
cd "$(dirname "$0")/.."
LOG=results/v7_trainval.log
DIR=results/v7_trainval
DONE_EPOCH=19        # EPOCHS=20 -> last epoch index 19

for attempt in $(seq 1 30); do
    if [ -f "$DIR/last-epoch.ckpt" ]; then
        echo "=== attempt $attempt: resuming from last-epoch.ckpt ===" >> "$LOG"
    else
        echo "=== attempt $attempt: fresh start ===" >> "$LOG"
        rm -rf "$DIR"
    fi
    EXP=v7_trainval TRAIN_ORIGIN_ONLY=true TRAIN_MODE=train_validation \
      NEW_HEAD_LR_MULT=5.0 EPOCHS=20 VAL_EVERY=5 CROP_LEN=4.0 CROP_MIN=30000 \
      LR=0.0002 ORIGIN_AGG=attn \
      bash scripts/train_arti_v2.sh >> "$LOG" 2>&1
    rc=$?
    # finished all epochs?
    last=$(tr '\r' '\n' < "$LOG" | grep -oE "Epoch ${DONE_EPOCH}: 100%" | tail -1)
    if [ -n "$last" ] || [ $rc -eq 0 ]; then
        echo "=== training complete (rc=$rc) ===" >> "$LOG"
        break
    fi
    echo "=== crashed rc=$rc, waiting 120s for GPU pressure to ease ===" >> "$LOG"
    sleep 120
done
echo V7_SUPERVISOR_DONE
