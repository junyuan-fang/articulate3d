#!/bin/bash
# Inter inference on the X-MIRRORED copies of the point clouds (flip TTA).
# v5/v6 were trained WITH mirror augmentation, so the flipped scene is
# in-distribution — unlike the mov-side TTA that failed on an unflipped
# backbone. Point order is preserved by the flip, so returned per-point masks
# align 1:1 with the original clouds and need no back-transform.
#   MODE=test|validation  CKPT=...  SAVE_DIR=...
set -e
cd "$(dirname "$0")/.."
source ./env.sh
export WANDB_MODE=offline
export WANDB_API_KEY=${WANDB_API_KEY:-dummy}

CKPT=${CKPT:?}
SAVE_DIR=${SAVE_DIR:?}
MODE=${MODE:-test}
FLIP=/data/challenge/scratch/inter_flip

python main_instance_segmentation_articulation.py \
    general.experiment_name="$(basename "$SAVE_DIR")" \
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
    data.test_mode="$MODE" \
    data.test_dataset.data_dir="$FLIP" \
    data.test_dataset.label_db_filepath="$FLIP/label_database.yaml" \
    data.validation_dataset.data_dir="$FLIP" \
    data.validation_dataset.label_db_filepath="$FLIP/label_database.yaml" \
    model.num_queries=100 \
    model.predict_articulation_mode=2 \
    model.predict_hierarchy_interaction=false \
    model.predict_articulation=false \
    loss.regular_arti_loss=false \
    loss.losses="[labels,masks]" \
    trainer.check_val_every_n_epoch=20 \
    optimizer.lr=0.0001

echo "preds: $SAVE_DIR/debug/val_preds/preds.pkl"
