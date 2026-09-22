#!/bin/bash
# Val inference with mask-side knobs exposed for the AP50 sweep (2026-08-24):
#   QUERIES  model.num_queries        (non-parametric FPS queries, safe to raise at test time)
#   TOPK     general.topk_per_image   (top-k over query x class pairs; -1 = keep all)
#   DBSCAN   general.use_dbscan       (split proposals into spatial clusters, CPU-heavy)
# Everything else identical to infer_mov_val_v2.sh gold settings.
set -e
cd "$(dirname "$0")/.."
source ./env.sh

export WANDB_MODE=offline
export WANDB_API_KEY=${WANDB_API_KEY:-dummy}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=${CKPT:-./results/v3_origin_bnfix/ep19.ckpt}
SAVE_DIR=${SAVE_DIR:?set SAVE_DIR}
ORIGIN_AGG=${ORIGIN_AGG:-attn}
AXIS_BIN=${AXIS_BIN:-true}
AXIS_RES_SCALE=${AXIS_RES_SCALE:-0.3}
QUERIES=${QUERIES:-100}
TOPK=${TOPK:-100}
DBSCAN=${DBSCAN:-false}

python eval_val.py \
    general.experiment_name="$(basename "$SAVE_DIR")" \
    general.project_name="articulate3d_challenge" \
    data/datasets=articulate3d_challenge_mov \
    general.num_targets=4 \
    general.eval_on_segments=false \
    general.train_on_segments=false \
    general.save_dir="$SAVE_DIR" \
    general.eval_articulation=true \
    general.eval_hierarchy_inter=false \
    general.debug=true \
    general.train_mode=false \
    general.checkpoint="$CKPT" \
    general.topk_per_image=$TOPK \
    general.use_dbscan=$DBSCAN \
    data.train_mode="train_validation" \
    data.num_labels=3 \
    data.batch_size=1 \
    data.voxel_size=0.02 \
    data.load_articulation=true \
    data.use_hierarchy=false \
    data.cropping=false \
    data.crop_length=8.0 \
    data.crop_min_size=100000 \
    data.use_coarse_to_fine=true \
    data.c2f_rad=0.1 \
    data.c2f_decay=0.4 \
    data.c2f_alpha=100 \
    data.test_mode="validation" \
    model.num_queries=$QUERIES \
    model.predict_articulation_mode=2 \
    model.predict_hierarchy_interaction=false \
    model.predict_articulation=true \
    model.origin_agg="$ORIGIN_AGG" \
    model.axis_bin_residual=$AXIS_BIN \
    model.axis_res_scale=$AXIS_RES_SCALE \
    loss.regular_arti_loss=false \
    loss.losses="[labels,masks, articulations]" \
    trainer.check_val_every_n_epoch=20 \
    optimizer.lr=0.0001

echo
echo "preds: $SAVE_DIR/debug/val_preds/preds.pkl"
