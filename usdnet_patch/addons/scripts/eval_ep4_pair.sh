#!/bin/bash
# Quantify the BatchNorm-buffer drift found in results/v2_origin_only @ep4.
# Same weights in both runs; the only difference is whether the 186 drifted BN
# running stats are kept or restored from mov_trainval.ckpt.
set -e
cd "$(dirname "$0")/.."

CKPT_DRIFT="./results/v2_origin_only/epoch=4-MAO_ST_ap50=0.3369.ckpt"
CKPT_RESTORED="./results/v2_origin_only/ep4_bnrestored.ckpt"

CKPT="$CKPT_DRIFT" SAVE_DIR=./results/eval_ep4_bndrift ORIGIN_AGG=attn AXIS_BIN=false \
    bash scripts/infer_mov_val_v2.sh > ./results/eval_ep4_bndrift.log 2>&1
echo "DONE bndrift"

CKPT="$CKPT_RESTORED" SAVE_DIR=./results/eval_ep4_bnrestored ORIGIN_AGG=attn AXIS_BIN=false \
    bash scripts/infer_mov_val_v2.sh > ./results/eval_ep4_bnrestored.log 2>&1
echo "DONE bnrestored"
echo ALLDONE
