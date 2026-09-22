#!/bin/bash
# External checkpoint snapshotter for v10_maskfix: the yaml ModelCheckpoint only
# fires on validation end, which never happens with VAL_EVERY=9999. Copy
# last-epoch.ckpt (rewritten every epoch) into snap/ on every 5th epoch.
LOG=/data/challenge/USDNet/results/v10_maskfix.log
CKPT=/data/challenge/USDNet/results/v10_maskfix/last-epoch.ckpt
SNAP=/data/challenge/USDNet/results/v10_maskfix/snap
last=-1
while true; do
  e=$(tr '\r' '\n' < "$LOG" | grep -oE "^Epoch [0-9]+: 100%" | tail -1 | grep -oE "[0-9]+" | head -1)
  if [ -n "$e" ] && [ "$e" != "$last" ] && [ $((e % 10)) -eq 9 ] && [ ! -f "$SNAP/epoch_$e.ckpt" ]; then
    s1=$(stat -c%s "$CKPT" 2>/dev/null); sleep 8; s2=$(stat -c%s "$CKPT" 2>/dev/null)
    if [ -n "$s1" ] && [ "$s1" = "$s2" ]; then
      cp "$CKPT" "$SNAP/epoch_$e.ckpt.tmp" && mv "$SNAP/epoch_$e.ckpt.tmp" "$SNAP/epoch_$e.ckpt"
      echo "snapped epoch $e"
      last=$e
    fi
  fi
  if tr '\r' '\n' < "$LOG" | grep -q "Trainer.fit. stopped"; then
    echo "training finished, snapshotter exiting"; break
  fi
  sleep 45
done
