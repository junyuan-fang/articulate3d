#!/bin/bash
# Final-shot pipeline: v7 (trainval origin, fixed 20-epoch budget, NO val
# selection -- val is training data for this checkpoint) + bin warm start +
# geometric stack -> verified submission zip.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis
PY=$HOME/.conda/envs/usdnet/bin/python

CKPT=${CKPT:-results/v7_trainval/last.ckpt}
[ -f "$CKPT" ] || { echo "no ckpt at $CKPT"; exit 1; }

# 1. test inference (bin head on, res zero-init -> scale irrelevant but pass 0.1)
rm -rf results/infer_test_v7 results/infer_test_v7.log
CKPT="$CKPT" SAVE_DIR=./results/infer_test_v7 ORIGIN_AGG=attn AXIS_BIN=true AXIS_RES_SCALE=0.1 \
  bash scripts/infer_mov_test_v2.sh > results/infer_test_v7.log 2>&1
P=results/infer_test_v7/debug/val_preds/preds.pkl
[ -f "$P" ] || { echo "inference produced no preds"; exit 1; }

# 2. geometric stack (PCA translation axis + endpoint origin with handle side)
$PY /data/challenge/analysis/postprocess2.py "$P" /data/challenge/analysis/test_v7_geo.pkl \
  data/processed/articulate3d_challenge_mov/test \
  --steps pca,endpoint \
  --handles results/inference_inter_seg/debug/val_preds/preds.pkl 2>/dev/null | tail -1

# 3. integrity checks
$PY - <<'EOF'
import pickle, numpy as np, json
a = pickle.load(open('results/infer_test_v7/debug/val_preds/preds.pkl', 'rb'))
b = pickle.load(open('/data/challenge/analysis/test_v7_geo.pkl', 'rb'))
sp = json.load(open('/data/challenge/data_download/articulate3d_splits.json'))
assert set(b) == set(sp['test']), "scene set mismatch"
assert all(np.array_equal(np.asarray(a[s]['pred_masks']), np.asarray(b[s]['pred_masks'])) for s in a), "masks changed"
n = np.concatenate([np.linalg.norm(np.asarray(v['pred_axises']), axis=1) for v in b.values()])
assert abs(n.min() - 1) < 1e-4 and abs(n.max() - 1) < 1e-4, "axes not unit"
c = np.concatenate([np.asarray(v['pred_classes']).ravel() for v in b.values()])
assert set(c.tolist()) <= {1, 2}
print("integrity: 42 scenes, masks untouched, axes unit, classes {1,2}  OK")
EOF

# 4. package
rm -rf results/sub_v7 && mkdir -p results/sub_v7
$PY transfer_preds_files.py --preds_pkl_dir /data/challenge/analysis/test_v7_geo.pkl \
  --preds_dir results/sub_v7/mov_test
cd results/sub_v7 && zip -rq mov_test.zip mov_test
echo "READY: results/sub_v7/mov_test.zip"
md5sum mov_test.zip
