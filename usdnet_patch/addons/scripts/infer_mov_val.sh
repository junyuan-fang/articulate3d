#!/bin/bash
# Same inference as infer_mov_local.sh but on the validation split, which we
# have ground truth for, so the trainer's eval path actually runs and prints
# metrics. `data.test_mode` selects which split the test dataloader reads, and
# trainer.py gates all evaluation on `test_mode != "test"`.
#
# CAVEAT: the only released checkpoint is mov_trainval.ckpt -- trained on
# train+validation. These numbers are therefore measured on data the model was
# fit on and will be optimistic. Use this to prove the pipeline and eval code
# work end to end, not as an estimate of test-set performance.
set -e
cd "$(dirname "$0")/.."
source ./env.sh

export WANDB_MODE=offline
export WANDB_API_KEY=${WANDB_API_KEY:-dummy}

CKPT=${CKPT:-/data/challenge/ckpts/checkpoints/mov_trainval.ckpt}
SAVE_DIR=${SAVE_DIR:-./results/eval_mov_val}
CURR_QUERY=100

python eval_val.py \
    general.experiment_name="eval_mov_val" \
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
    data.test_mode="validation" \
    model.num_queries=${CURR_QUERY} \
    model.predict_articulation_mode=2 \
    model.predict_hierarchy_interaction=false \
    model.predict_articulation=true \
    loss.regular_arti_loss=false \
    loss.losses="[labels,masks, articulations]" \
    trainer.check_val_every_n_epoch=20 \
    optimizer.lr=0.0001
