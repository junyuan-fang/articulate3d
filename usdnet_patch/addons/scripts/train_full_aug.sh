#!/bin/bash
# v9 -- the one experiment the probe says is necessary: UNFREEZE everything and
# retrain with real geometric augmentation.
#
# Rationale: a linear probe on the frozen 256-d axis-head input separates
# lateral from vertical hinges at AUC 0.247 (permutation p=0.99) -- the
# information simply is not in those features. No adapter, head capacity or
# schedule downstream of them can fix that; only changing the features can.
#
# Augmentation: random yaw (GT axes/origins rotated with the points, verified
# invariant by analysis/test_rot_aug.py) + mirror flip (component-wise axis
# transform, bug fixed earlier). This is the first training run in this project
# where the backbone sees more than one pose per scene.
#
# WARNING: this deliberately gives up the bit-identical-AP50 guarantee that
# every previous run held. Mask quality is our only leaderboard-leading number,
# so evaluate before trusting anything from here.
#
# 2026-08-24: in-training validation OOMs at 24 GB -- full-network AdamW state
# (~19 GB) + whole-scene val inference (>9 GB) cannot coexist. Validation is
# disabled (VAL_EVERY=9999) and checkpoints are saved unconditionally every
# SAVE_EVERY epochs; select offline with scripts/infer_mov_val_sweep.sh after.
set -e
cd "$(dirname "$0")/.."
source ./env.sh

export WANDB_MODE=offline
export WANDB_API_KEY=${WANDB_API_KEY:-dummy}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP=${EXP:-v9_fullaug}
CKPT=${CKPT:-/data/challenge/ckpts/checkpoints/mov_trainval.ckpt}
SAVE_DIR=${SAVE_DIR:-./results/$EXP}
EPOCHS=${EPOCHS:-120}
VAL_EVERY=${VAL_EVERY:-5}
LR=${LR:-0.00005}          # 4x below the head-only runs: full network, small data

python main_instance_segmentation_articulation.py \
    general.experiment_name="$EXP" \
    general.project_name="articulate3d_challenge" \
    data/datasets=articulate3d_challenge_mov \
    callbacks=callbacks_articulation \
    general.num_targets=4 \
    general.eval_on_segments=false \
    general.train_on_segments=false \
    general.save_dir="$SAVE_DIR" \
    general.eval_articulation=true \
    general.eval_hierarchy_inter=false \
    general.train_mode=true \
    general.freeze_except_articulation=false \
    general.checkpoint="$CKPT" \
    data.train_mode="${TRAIN_MODE:-train}" \
    data.num_labels=3 \
    data.batch_size=1 \
    data.voxel_size=0.02 \
    data.load_articulation=true \
    data.use_hierarchy=false \
    data.cropping=true \
    data.crop_length=${CROP_LEN:-4.0} \
    data.crop_min_size=${CROP_MIN:-30000} \
    data.use_coarse_to_fine=${C2F:-true} \
    data.c2f_rad=0.1 \
    data.c2f_decay=0.4 \
    data.c2f_alpha=100 \
    data.train_dataset.random_flip=true \
    data.train_dataset.random_rot_z=${ROT_Z:-3.14159} \
    model.num_queries=100 \
    model.predict_articulation_mode=2 \
    model.predict_hierarchy_interaction=false \
    model.predict_articulation=true \
    model.origin_agg=${ORIGIN_AGG:-attn} \
    model.axis_bin_residual=${AXIS_BIN:-false} \
    loss.regular_arti_loss=false \
    loss.axis_loss_abs_cos=true \
    loss.losses="[labels,masks, articulations]" \
    trainer.max_epochs=${EPOCHS} \
    trainer.check_val_every_n_epoch=${VAL_EVERY} \
    callbacks.0.monitor=null \
    callbacks.0.save_top_k=-1 \
    callbacks.0.every_n_epochs=${SAVE_EVERY:-5} \
    optimizer.lr=${LR}
