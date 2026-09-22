#!/bin/bash
LOG=/data/challenge/USDNet/results/inter_v5_tv.log
CKPT=/data/challenge/USDNet/results/inter_v5_tv/last-epoch.ckpt
SNAP=/data/challenge/USDNet/results/inter_v5_tv/snap
mkdir -p "$SNAP"; last=-1
while true; do
  e=$(tr '\r' '\n' < "$LOG" | grep -oE "^Epoch [0-9]+: 100%" | tail -1 | grep -oE "[0-9]+" | head -1)
  if [ -n "$e" ] && [ "$e" != "$last" ] && [ $((e % 5)) -eq 4 ] && [ ! -f "$SNAP/epoch_$e.ckpt" ]; then
    s1=$(stat -c%s "$CKPT" 2>/dev/null); sleep 5; s2=$(stat -c%s "$CKPT" 2>/dev/null)
    if [ -n "$s1" ] && [ "$s1" = "$s2" ]; then
      cp "$CKPT" "$SNAP/epoch_$e.ckpt.tmp" && mv "$SNAP/epoch_$e.ckpt.tmp" "$SNAP/epoch_$e.ckpt"
      echo "snapped epoch $e"; last=$e
    fi
  fi
  tr '\r' '\n' < "$LOG" | grep -q "Trainer.fit. stopped" && { echo "done"; break; }
  sleep 20
done
