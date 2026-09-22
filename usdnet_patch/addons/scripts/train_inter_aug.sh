#!/bin/bash
# inter_v2 -- retrain the interactable-part segmentation model with real
# geometric augmentation (mirror flip + random yaw), same rationale as
# train_full_aug.sh: the released inter_trainval.ckpt saw one pose per scene.
# Goal: beat ZZZ's inter AP50 (test 0.2829 vs our baseline 0.2686).
#
# The inter dataset yaml has no random_flip/random_rot_z keys, hence the
# +data.train_dataset.* injections (mov yaml declares them, inter never did).
set -e
cd "$(dirname "$0")/.."
source ./env.sh

export WANDB_MODE=offline
export WANDB_API_KEY=${WANDB_API_KEY:-dummy}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP=${EXP:-inter_v2_aug}
CKPT=${CKPT:-/data/challenge/ckpts/checkpoints/inter_trainval.ckpt}
SAVE_DIR=${SAVE_DIR:-./results/$EXP}
EPOCHS=${EPOCHS:-60}
VAL_EVERY=${VAL_EVERY:-9999}   # in-training whole-scene val OOMs at 24GB (v9 lesson)
LR=${LR:-0.00001}          # v9 lesson: aggressive aug+LR destroys masks; go gentle

python main_instance_segmentation_articulation.py \
    general.experiment_name="$EXP" \
    general.project_name="articulate3d_challenge" \
    data/datasets=articulate3d_challenge_inter \
    general.num_targets=4 \
    general.eval_on_segments=false \
    general.train_on_segments=false \
    general.save_dir="$SAVE_DIR" \
    general.eval_articulation=false \
    general.eval_hierarchy_inter=false \
    general.train_mode=true \
    general.freeze_except_articulation=false \
    general.checkpoint="$CKPT" \
    data.train_mode="${TRAIN_MODE:-train}" \
    data.num_labels=3 \
    data.batch_size=1 \
    data.voxel_size=0.02 \
    data.load_articulation=false \
    data.use_hierarchy=false \
    data.cropping=${CROPPING:-false} \
    data.crop_length=8.0 \
    data.crop_min_size=100000 \
    data.use_coarse_to_fine=${C2F:-false} \
    data.c2f_rad=0.1 \
    data.c2f_decay=0.4 \
    data.c2f_alpha=100 \
    +data.train_dataset.random_flip=true \
    +data.train_dataset.random_rot_z=${ROT_Z:-0.0} \
    model.num_queries=100 \
    model.predict_articulation_mode=2 \
    model.predict_hierarchy_interaction=false \
    model.predict_articulation=false \
    loss.regular_arti_loss=false \
    loss.losses="[labels,masks]" \
    trainer.max_epochs=${EPOCHS} \
    trainer.check_val_every_n_epoch=${VAL_EVERY} \
    optimizer.lr=${LR}
