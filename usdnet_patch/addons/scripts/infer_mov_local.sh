#!/bin/bash
# Movable-part segmentation + articulation inference, wired up for env/env.sh.
#
# Differences from scripts/infer_mov.sh:
#   - real checkpoint path and no `cd USDNet` (run it from the repo root)
#   - wandb offline: the original needs an API key, and inference doesn't need
#     a run logged anywhere
#   - CUDA_LAUNCH_BLOCKING left off; the original sets it for debugging and it
#     costs real throughput.
#
# The transfer + zip steps ARE kept: the submission server wants the ScanNet
# instance-segmentation zip layout, which transfer_preds_files.py writes. (The
# GitHub-pages challenge.html says the opposite -- submit preds.pkl, don't run
# the transfer -- but the server's own Submission Guidelines and these scripts
# agree on the zip, so the page is stale.)
set -e
cd "$(dirname "$0")/.."
source ./env.sh

export WANDB_MODE=offline
export WANDB_API_KEY=${WANDB_API_KEY:-dummy}

CKPT=${CKPT:-/data/challenge/ckpts/checkpoints/mov_trainval.ckpt}
SAVE_DIR=${SAVE_DIR:-./results/inference_mov_articulation}
CURR_QUERY=100

if [ ! -f "$CKPT" ]; then
    echo "checkpoint not found: $CKPT" >&2
    exit 1
fi

python main_instance_segmentation_articulation.py \
    general.experiment_name="inference_mov_articulation" \
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
    model.num_queries=${CURR_QUERY} \
    model.predict_articulation_mode=2 \
    model.predict_hierarchy_interaction=false \
    model.predict_articulation=true \
    loss.regular_arti_loss=false \
    loss.losses="[labels,masks, articulations]" \
    trainer.check_val_every_n_epoch=20 \
    optimizer.lr=0.0001

echo
echo "submission file: $SAVE_DIR/debug/val_preds/preds.pkl"
echo "submit that .pkl directly -- do not run transfer_preds_files.py or zip it."
