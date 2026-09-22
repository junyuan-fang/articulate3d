#!/bin/bash
# Interactable-part segmentation inference on the TEST split -> preds.pkl.
#
# Derived from scripts/infer_inter.sh with the two TODO placeholders filled in
# (BACKBONE_CKPT, and the `cd USDNet` replaced by cd-to-repo-root) plus:
#   - wandb offline (the original wants an API key; inference needs no logging)
#   - CUDA_LAUNCH_BLOCKING left off (the original sets it for debugging, it
#     costs real throughput)
#   - transfer + zip left out; done explicitly afterwards so the output dir can
#     be wiped first (transfer_preds_files.py appends to <scene>.txt).
#
# `set -e` only -- `set -u` breaks conda's cuda-nvcc activation hook.
set -e
cd "$(dirname "$0")/.."
source ./env.sh

export WANDB_MODE=offline
export WANDB_API_KEY=${WANDB_API_KEY:-dummy}

CKPT=${CKPT:-/data/challenge/ckpts/checkpoints/inter_trainval.ckpt}
SAVE_DIR=${SAVE_DIR:-./results/inference_inter_seg}
CURR_QUERY=100

if [ ! -f "$CKPT" ]; then
    echo "checkpoint not found: $CKPT" >&2
    exit 1
fi

python main_instance_segmentation_articulation.py \
    general.experiment_name="inference_inter_seg" \
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
    data.test_mode="test" \
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
