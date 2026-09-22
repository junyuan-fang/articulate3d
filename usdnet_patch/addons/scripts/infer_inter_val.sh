#!/bin/bash
# Interactable-part segmentation on the VALIDATION split -> preds.pkl + local AP.
#
# Uses eval_val.py (runner.validate) rather than main_...py (runner.test):
# eval_instance_epoch_end gates the AP block behind `mode != 'test'`, so the
# test path would never print metrics. validate() also writes preds.pkl (the
# `if self.debug:` dump sits outside that gate), so one run gives us both the
# submission input and a local sanity number.
#
# validate() reads `validation_dataset` (mode=${data.validation_mode}), while
# data.test_mode="validation" is what unlocks the eval branches in trainer.py.
# That also dodges a typo in conf/data/datasets/articulate3d_challenge_inter.yaml:92
# (`load_articulation: ${data.load_articulation},` -- the trailing comma turns
# the interpolation into the *string* "False," which is truthy). It only bites
# test_dataset, and there mode=="test" forces load_articulation off anyway.
#
# NOTE: trainer.py:560 drops scenes whose GT is empty when test_mode != "test",
# so val yields 41 of 42 scenes. Known and accepted by the server.
set -e
cd "$(dirname "$0")/.."
source ./env.sh

export WANDB_MODE=offline
export WANDB_API_KEY=${WANDB_API_KEY:-dummy}

CKPT=${CKPT:-/data/challenge/ckpts/checkpoints/inter_trainval.ckpt}
SAVE_DIR=${SAVE_DIR:-./results/inference_inter_seg_val}
CURR_QUERY=100

if [ ! -f "$CKPT" ]; then
    echo "checkpoint not found: $CKPT" >&2
    exit 1
fi

python eval_val.py \
    general.experiment_name="inference_inter_seg_val" \
    general.project_name="articulate3d_challenge" \
    data/datasets=articulate3d_challenge_inter \
    general.num_targets=4 \
    general.eval_on_segments=false \
    general.train_on_segments=false \
    general.save_dir="$SAVE_DIR" \
    general.eval_articulation=false \
    general.eval_hierarchy_inter=false \
    general.debug=true \
    general.train_mode=false \
    general.checkpoint="$CKPT" \
    data.train_mode="train_validation" \
    data.num_labels=3 \
    data.batch_size=1 \
    data.voxel_size=0.02 \
    data.load_articulation=false \
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
    model.predict_articulation=false \
    loss.regular_arti_loss=false \
    loss.losses="[labels,masks]" \
    trainer.check_val_every_n_epoch=20 \
    optimizer.lr=0.0001

echo
echo "preds: $SAVE_DIR/debug/val_preds/preds.pkl"
