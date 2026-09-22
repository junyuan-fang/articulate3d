#!/bin/bash
# Flip-TTA: merge original + mirrored val predictions, score raw and geo-stacked.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis
PY=$HOME/.conda/envs/usdnet/bin/python
D=data/processed/articulate3d_challenge_mov/validation
CONSTS=data/processed/articulate3d_challenge_mov_flipx/flip_constants.json
HANDLES=results/inference_inter_seg_val/debug/val_preds/preds.pkl

$PY /data/challenge/analysis/merge_flip_tta.py \
  results/ev_v4init/debug/val_preds/preds.pkl \
  results/ev_flipx/debug/val_preds/preds.pkl \
  /tmp/val_tta.pkl --consts $CONSTS 2>/dev/null

echo "=== merged (200q, no NMS) raw ==="
$PY /data/challenge/analysis/score_pkl.py /tmp/val_tta.pkl 2>/dev/null | grep AP50

$PY /data/challenge/analysis/postprocess2.py /tmp/val_tta.pkl /tmp/val_tta_geo.pkl $D \
  --steps pca,endpoint --handles $HANDLES 2>/dev/null | tail -1
echo "=== merged + geo stack ==="
$PY /data/challenge/analysis/score_pkl.py /tmp/val_tta_geo.pkl 2>/dev/null | grep AP50

$PY /data/challenge/analysis/merge_flip_tta.py \
  results/ev_v4init/debug/val_preds/preds.pkl \
  results/ev_flipx/debug/val_preds/preds.pkl \
  /tmp/val_tta_nms.pkl --consts $CONSTS --nms 0.8 2>/dev/null
$PY /data/challenge/analysis/postprocess2.py /tmp/val_tta_nms.pkl /tmp/val_tta_nms_geo.pkl $D \
  --steps pca,endpoint --handles $HANDLES 2>/dev/null | tail -1
echo "=== merged + NMS(0.8) + geo stack ==="
$PY /data/challenge/analysis/score_pkl.py /tmp/val_tta_nms_geo.pkl 2>/dev/null | grep AP50
