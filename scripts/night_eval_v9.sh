#!/bin/bash
# After v9 training completes: evaluate snapshot checkpoints on val with gold
# inference settings; print mask AP50 + raw joint (MAO_ST) per snapshot.
cd /data/challenge/USDNet
LOG=results/v9_fullaug.log
until tr '\r' '\n' < "$LOG" | grep -q "Trainer.fit. stopped"; do sleep 120; done
echo "=== training done, starting snapshot evals $(date '+%H:%M') ==="
sleep 60   # let teardown release GPU
for CK in results/v9_fullaug/snap/epoch_59.ckpt results/v9_fullaug/snap/epoch_79.ckpt \
          results/v9_fullaug/snap/epoch_94.ckpt results/v9_fullaug/snap/epoch_104.ckpt \
          results/v9_fullaug/snap/epoch_114.ckpt results/v9_fullaug/last-epoch.ckpt; do
  [ -f "$CK" ] || { echo "MISSING $CK"; continue; }
  N=$(basename "$CK" .ckpt)
  SD=./results/v9ev_$N
  rm -rf "$SD" "results/v9ev_$N.log"
  CKPT="$CK" SAVE_DIR="$SD" ORIGIN_AGG=attn AXIS_BIN=true AXIS_RES_SCALE=0.3 \
    bash scripts/infer_mov_val_sweep.sh > "results/v9ev_$N.log" 2>&1
  M=$(tr '\r' '\n' < "results/v9ev_$N.log" | grep -A6 "^what_  " | grep average | awk '{print $4}')
  J=$(tr '\r' '\n' < "results/v9ev_$N.log" | grep -A6 "^what_MAO_ST" | grep average | awk '{print $4}')
  echo "SNAP $N mask_AP50=$M raw_joint_AP50=$J"
  # free disk: keep pkl only for now-best handling later; drop wandb
  rm -rf "$SD/wandb"
done
echo "=== snapshot evals done $(date '+%H:%M') ==="
