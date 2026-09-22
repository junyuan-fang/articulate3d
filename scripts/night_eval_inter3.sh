#!/bin/bash
cd /data/challenge/USDNet
LOG=results/inter_v3_sharp.log
until tr '\r' '\n' < "$LOG" | grep -q "Trainer.fit. stopped"; do sleep 60; done
echo "=== inter training done, evals start $(date '+%H:%M') ==="
sleep 45
for CK in results/inter_v3_sharp/snap/epoch_19.ckpt results/inter_v3_sharp/snap/epoch_29.ckpt \
          results/inter_v3_sharp/snap/epoch_39.ckpt results/inter_v3_sharp/snap/epoch_49.ckpt \
          results/inter_v3_sharp/snap/epoch_59.ckpt results/inter_v3_sharp/last-epoch.ckpt; do
  [ -f "$CK" ] || { echo "MISSING $CK"; continue; }
  N=$(basename "$CK" .ckpt)
  SD=./results/inter3ev_$N
  rm -rf "$SD" "results/inter3ev_$N.log"
  CKPT="$CK" SAVE_DIR="$SD" bash scripts/infer_inter_val.sh > "results/inter3ev_$N.log" 2>&1
  A=$(tr '\r' '\n' < "results/inter3ev_$N.log" | grep -A6 "^what_  " | grep average | awk '{print $4}')
  echo "SNAP $N val_AP50=$A"
  rm -rf "$SD/wandb"
done
echo "=== inter evals done $(date '+%H:%M') ==="
