#!/bin/bash
cd /data/challenge/USDNet
LOG=results/inter_v2_aug.log
until tr '\r' '\n' < "$LOG" | grep -q "Trainer.fit. stopped"; do sleep 60; done
echo "=== inter training done, evals start $(date '+%H:%M') ==="
sleep 45
for CK in results/inter_v2_aug/snap/epoch_19.ckpt results/inter_v2_aug/snap/epoch_29.ckpt \
          results/inter_v2_aug/snap/epoch_39.ckpt results/inter_v2_aug/snap/epoch_49.ckpt \
          results/inter_v2_aug/snap/epoch_59.ckpt results/inter_v2_aug/last-epoch.ckpt; do
  [ -f "$CK" ] || { echo "MISSING $CK"; continue; }
  N=$(basename "$CK" .ckpt)
  SD=./results/interev_$N
  rm -rf "$SD" "results/interev_$N.log"
  CKPT="$CK" SAVE_DIR="$SD" bash scripts/infer_inter_val.sh > "results/interev_$N.log" 2>&1
  A=$(tr '\r' '\n' < "results/interev_$N.log" | grep -A6 "^what_  " | grep average | awk '{print $4}')
  echo "SNAP $N val_AP50=$A"
  rm -rf "$SD/wandb"
done
echo "=== inter evals done $(date '+%H:%M') ==="
