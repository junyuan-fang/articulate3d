#!/bin/bash
# Score one checkpoint on val, raw and with the (corrected) training-free
# post-processing, using the official criteria. Local val reproduces the server
# bit for bit, so this spends no submission quota.
#
#   bash eval_ckpt.sh <ckpt> <tag> [ORIGIN_AGG] [AXIS_BIN]
#
# ORIGIN_AGG / AXIS_BIN default to attn/true. Setting them to mean/false while
# pointing at a v2 checkpoint is the ablation: the trained head is bypassed at
# inference and the stock path runs instead.
set -e
CKPT=$1
TAG=$2
AGG=${3:-attn}
BIN=${4:-true}
OUT=/data/challenge/USDNet/results/eval_$TAG

source /data/challenge/USDNet/env.sh
export PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis

rm -rf "$OUT"
CKPT="$CKPT" SAVE_DIR="$OUT" ORIGIN_AGG="$AGG" AXIS_BIN="$BIN" \
    bash /data/challenge/USDNet/scripts/infer_mov_val_v2.sh \
    > "/data/challenge/USDNet/results/eval_$TAG.log" 2>&1

PKL="$OUT/debug/val_preds/preds.pkl"
cd /data/challenge/analysis
echo "=== $TAG  (origin_agg=$AGG axis_bin=$BIN) ==="
echo "--- raw ---"
python score_pkl.py "$PKL" 2>&1 | grep -E "AP50"
echo "--- + post-processing sweep ---"
python pp_sweep.py "$PKL" 2>&1 | grep -E "AP50 "
