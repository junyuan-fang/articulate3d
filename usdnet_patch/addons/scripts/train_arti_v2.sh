#!/bin/bash
# Articulation head v2 finetune -- see docs/06-finetune-design.md.
#
# What changes and why (evidence in docs/04-diagnosis.md):
#
#   origin_agg=attn
#       The stock mode-2 origin head averages (point + offset) uniformly over the
#       whole predicted mask, and a uniform average of a part is its centroid.
#       77.2% of predicted origins are dragged toward the centroid and recover
#       only 29% of the centroid->hinge distance. An oracle that is allowed to
#       pick the single best point *of the model's own predicted mask* scores
#       100% under the 0.25 m criterion with a 6 mm median error (val AP50_origin
#       0.4495 -> 0.5473, joint 0.3215 -> 0.4439), so the origin problem is
#       entirely a point-selection problem. A query-conditioned masked softmax
#       replaces the mean with a learnable selection; zero-initialised, it starts
#       numerically at the mean (analysis/test_head_v2_identity.py).
#
#   axis_bin_residual=true   (ABLATION -- set AXIS_BIN=true to enable)
#       93.3% of matched rotation GT axes are within 15 deg of gravity, so free
#       3-vector regression collapses onto z: all 8 lateral hinges fail by 85-90
#       deg and the trivial "always z" baseline beats the model 9 losses to 0
#       wins. Discrete anchor choice + in-bin residual removes the averaged
#       compromise direction. Warm-started from the free branch.
#
#   freeze_except_articulation=true
#       Our AP50 is the highest on the leaderboard; nothing upstream of the
#       articulation heads is trained, so AP50 provably cannot move.
#
#   axis_loss_abs_cos=true
#       The metric scores |cos|, so a 180 deg flip is a perfect hit; the stock
#       1-cos loss gives it maximum penalty.
#
#   callbacks=callbacks_articulation
#       Select checkpoints on MAO_ST_ap50 (the leaderboard metric) rather than
#       val_mean_ap_50, which is constant once the backbone is frozen.
#
# Validate with scripts/infer_mov_val.sh + analysis/score_pkl.py; local val
# reproduces the server bit for bit, so no submission quota is spent.
set -e
cd "$(dirname "$0")/.."
source ./env.sh

export WANDB_MODE=offline
export WANDB_API_KEY=${WANDB_API_KEY:-dummy}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP=${EXP:-train_arti_v2}
CKPT=${CKPT:-/data/challenge/ckpts/checkpoints/mov_trainval.ckpt}
SAVE_DIR=${SAVE_DIR:-./results/$EXP}
# "train" = clean(er) generalisation estimate on val
# "train_validation" = the config to use for a final test submission
TRAIN_MODE=${TRAIN_MODE:-train}
EPOCHS=${EPOCHS:-150}
ORIGIN_AGG=${ORIGIN_AGG:-attn}
AXIS_BIN=${AXIS_BIN:-false}
LR=${LR:-0.0002}
VAL_EVERY=${VAL_EVERY:-10}

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
    general.freeze_except_articulation=true \
    general.new_head_lr_mult=${NEW_HEAD_LR_MULT:-1.0} \
    general.train_origin_only=${TRAIN_ORIGIN_ONLY:-false} \
    general.checkpoint="$CKPT" \
    data.train_mode="$TRAIN_MODE" \
    data.num_labels=3 \
    data.batch_size=1 \
    data.voxel_size=0.02 \
    data.load_articulation=true \
    data.use_hierarchy=false \
    data.cropping=true \
    data.crop_length=${CROP_LEN:-4.0} \
    data.crop_min_size=${CROP_MIN:-30000} \
    data.use_coarse_to_fine=true \
    data.c2f_rad=0.1 \
    data.c2f_decay=0.4 \
    data.c2f_alpha=100 \
    model.num_queries=100 \
    model.predict_articulation_mode=2 \
    model.predict_hierarchy_interaction=false \
    model.predict_articulation=true \
    model.origin_agg="$ORIGIN_AGG" \
    model.axis_bin_residual=$AXIS_BIN \
    model.axis_res_scale=${AXIS_RES_SCALE:-0.0} \
    loss.regular_arti_loss=false \
    loss.axis_loss_abs_cos=true \
    loss.origin_loss_sym=${ORIGIN_SYM:-true} \
    loss.losses="[labels,masks, articulations]" \
    trainer.max_epochs=${EPOCHS} \
    trainer.check_val_every_n_epoch=${VAL_EVERY} \
    optimizer.lr=$LR
