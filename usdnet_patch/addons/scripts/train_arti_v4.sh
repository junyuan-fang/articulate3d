#!/bin/bash
# v4 -- everything the research said, in one clean run (docs/04, 06, 07):
#
#   init            v3 ep19 (best origin-attn ckpt, BN-clean, val both 0.3477)
#   trainable       origin heads + axis_cls/axis_res only (train_heads_v2);
#                   free-regression axis heads stay frozen so the bin anchors'
#                   input never drifts and the zero-init warm start equals the
#                   0.4677 canonical-snap rule at load time
#   axis head       bin+residual, bounded (axis_res_scale>0). The BN-drift-era
#                   "harmful" verdict used an UNBOUNDED residual and is void.
#   loss            axis_loss_abs_cos=true (metric uses |cos|);
#                   axis_horiz_weight=5 (82/1285 lateral hinges, 14:1 -> ~1:3)
#   data            hbal database (44 lateral-hinge scenes x3, share 22.6->46.6%)
#                   + random_flip (component-wise axis transform fixed and
#                   verified: dist / |cos(axis,dir)| / |axis_z| all preserved)
#   BN              frozen in eval() -- v2's entire "gain" was BN drift
#
# Val is bit-identical to the server; do NOT submit from here (quota 8).
set -e
cd "$(dirname "$0")/.."
source ./env.sh

export WANDB_MODE=offline
export WANDB_API_KEY=${WANDB_API_KEY:-dummy}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP=${EXP:-v4_axis_origin}
CKPT=${CKPT:-./results/v3_origin_bnfix/ep19.ckpt}
SAVE_DIR=${SAVE_DIR:-./results/$EXP}
EPOCHS=${EPOCHS:-150}
VAL_EVERY=${VAL_EVERY:-5}

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
    general.train_heads_v2=${TRAIN_HEADS_V2:-true} \
    general.train_axis_res_only=${TRAIN_RES_ONLY:-false} \
    general.train_axis_cls_only=${TRAIN_CLS_ONLY:-false} \
    general.new_head_lr_mult=${NEW_HEAD_LR_MULT:-5.0} \
    general.checkpoint="$CKPT" \
    data.train_mode="train" \
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
    data.train_dataset.data_dir=${TRAIN_DB:-data/processed/articulate3d_challenge_mov_hbal} \
    data.train_dataset.random_flip=true \
    model.num_queries=100 \
    model.predict_articulation_mode=2 \
    model.predict_hierarchy_interaction=false \
    model.predict_articulation=true \
    model.origin_agg=attn \
    model.axis_bin_residual=true \
    model.axis_res_scale=${AXIS_RES_SCALE:-0.3} \
    loss.regular_arti_loss=false \
    loss.axis_loss_abs_cos=true \
    loss.axis_horiz_weight=${AXIS_HORIZ_WEIGHT:-5.0} \
    loss.losses="[labels,masks, articulations]" \
    trainer.max_epochs=${EPOCHS} \
    trainer.check_val_every_n_epoch=${VAL_EVERY} \
    optimizer.lr=${LR:-0.0002}
