#!/bin/bash
# Test-split inference for the v2 articulation heads (origin-attn + axis-bin).
# Same as infer_mov_local.sh but with the v2 model overrides and no
# transfer/zip tail -- packaging is a separate, verifiable step.
#
#   CKPT=...          checkpoint (default: v3 ep19, the "init-state" model)
#   SAVE_DIR=...      output dir; preds land in $SAVE_DIR/debug/val_preds/preds.pkl
#   ORIGIN_AGG=attn|mean   AXIS_BIN=true|false
set -e
cd "$(dirname "$0")/.."
source ./env.sh

export WANDB_MODE=offline
export WANDB_API_KEY=${WANDB_API_KEY:-dummy}

CKPT=${CKPT:-./results/v3_origin_bnfix/ep19.ckpt}
SAVE_DIR=${SAVE_DIR:-./results/infer_test_v2}
ORIGIN_AGG=${ORIGIN_AGG:-attn}
AXIS_BIN=${AXIS_BIN:-true}

python main_instance_segmentation_articulation.py \
    general.experiment_name="$(basename $SAVE_DIR)" \
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
    data.test_mode="test" \
    model.num_queries=100 \
    model.predict_articulation_mode=2 \
    model.predict_hierarchy_interaction=false \
    model.predict_articulation=true \
    model.origin_agg="$ORIGIN_AGG" \
    model.axis_bin_residual=$AXIS_BIN \
    model.axis_res_scale=${AXIS_RES_SCALE:-0.3} \
    loss.regular_arti_loss=false \
    loss.losses="[labels,masks, articulations]" \
    trainer.check_val_every_n_epoch=20 \
    optimizer.lr=0.0001

echo "preds: $SAVE_DIR/debug/val_preds/preds.pkl"
