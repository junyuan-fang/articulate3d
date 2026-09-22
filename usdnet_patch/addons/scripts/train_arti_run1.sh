#!/bin/bash
# Run 1 -- retrain ONLY the articulation heads, with the loss aligned to the
# evaluation criteria.
#
# Rationale (docs/04-diagnosis.md, docs/05-improvement.md):
#   - our AP50 (0.4370) is already the highest on the leaderboard, so the
#     segmentation branch is frozen and AP50 provably cannot move
#   - all the lost points are in articulation: rotation keeps only 66% of AP50
#     through the origin criterion, translation only 72.5% through the axis one
#   - two loss/metric misalignments are corrected here:
#       axis_loss_abs_cos=true   metric scores |cos|, so a 180-deg flip is a hit;
#                                the stock 1-cos loss gives it maximum penalty
#       origin_loss_sym=false    drops dist_pred_axis, a term the metric never
#                                measures and which pulls the gradient sideways
#
# Starts from the released mov_trainval.ckpt so the heads begin at baseline
# quality rather than from scratch.
#
# Validate with scripts/infer_mov_val.sh -- local val reproduces the server
# bit-for-bit, so no submission quota is spent.
set -e
cd "$(dirname "$0")/.."
source ./env.sh

export WANDB_MODE=offline
export WANDB_API_KEY=${WANDB_API_KEY:-dummy}

# The mode-2 articulation head materialises (batch, n_points, n_queries, 3)
# tensors -- ~120 MB each at 100k points x 100 queries -- and training keeps
# them alive for backward. Stock crop settings OOM on a 24 GB card, so crop
# smaller and let the allocator defragment (the failure showed 5.4 GB reserved
# but unallocated).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=${CKPT:-/data/challenge/ckpts/checkpoints/mov_trainval.ckpt}
SAVE_DIR=${SAVE_DIR:-./results/train_arti_run1}
# "train" = clean generalisation estimate on val.
# "train_validation" = the config to use for a final test submission.
TRAIN_MODE=${TRAIN_MODE:-train}
EPOCHS=${EPOCHS:-120}
CURR_QUERY=100

python main_instance_segmentation_articulation.py \
    general.experiment_name="train_arti_run1" \
    general.project_name="articulate3d_challenge" \
    data/datasets=articulate3d_challenge_mov \
    general.num_targets=4 \
    general.eval_on_segments=false \
    general.train_on_segments=false \
    general.save_dir="$SAVE_DIR" \
    general.eval_articulation=true \
    general.eval_hierarchy_inter=false \
    general.train_mode=true \
    general.freeze_except_articulation=true \
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
    model.num_queries=${CURR_QUERY} \
    model.predict_articulation_mode=2 \
    model.predict_hierarchy_interaction=false \
    model.predict_articulation=true \
    loss.regular_arti_loss=false \
    loss.axis_loss_abs_cos=true \
    loss.origin_loss_sym=false \
    loss.losses="[labels,masks, articulations]" \
    trainer.max_epochs=${EPOCHS} \
    trainer.check_val_every_n_epoch=10 \
    optimizer.lr=0.0001
