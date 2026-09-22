#!/bin/bash
# Re-run the geometric stack and repackage from the EXISTING v7 test
# predictions. The sliding-leaf rule lives entirely in post-processing, so the
# network output is unchanged and inference does not need to run again.
#
# The pickles here are ~5 GB each. Loading three at once has killed this script
# mid-run, so each step below opens only what it needs and drops it before the
# next. Do not merge the python blocks back together.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis
PY=$HOME/.conda/envs/usdnet/bin/python

RAW=results/infer_test_v7/debug/val_preds/preds.pkl
OLD=/data/challenge/analysis/test_v7_geo.pkl
OUT=/data/challenge/analysis/test_v8_slider.pkl
SUB=results/sub_v8
[ -f "$RAW" ] || { echo "no test preds at $RAW -- run endgame_package.sh instead"; exit 1; }

echo "[1/4] post-processing"
$PY /data/challenge/analysis/postprocess2.py "$RAW" "$OUT" \
  data/processed/articulate3d_challenge_mov/test \
  --steps pca,endpoint \
  --handles results/inference_inter_seg/debug/val_preds/preds.pkl 2>/dev/null | tail -1

echo "[2/4] what the leaf rule changed"
# Count over ALL predictions, not just confident ones. Restricting to score>0.3
# once reported "3 axes changed" for a run that actually rotated 50 -- and AP
# integrates the whole precision-recall curve, so the low-confidence tail counts.
# The shortest flipped leaf is the number to watch: a sub-0.45 m "sliding door"
# means MIN_LEAF_H is not doing its job and drawers are being rotated.
OLD="$OLD" OUT="$OUT" $PY - <<'PYEOF'
import os, pickle, numpy as np
old = pickle.load(open(os.environ['OLD'], 'rb'))
new = pickle.load(open(os.environ['OUT'], 'rb'))
tot = 0; hmin = 9e9; top = []
for s in new:
    a = np.asarray(old[s]['pred_axises'], float); b = np.asarray(new[s]['pred_axises'], float)
    d = np.degrees(np.arccos(np.clip(np.abs((a * b).sum(1)), 0, 1)))
    idx = np.where(d > 1)[0]
    if not len(idx):
        continue
    tot += len(idx)
    pts = np.asarray(np.load(f'data/processed/articulate3d_challenge_mov/test/{s}.npy',
                             mmap_mode='r')[:, 0:3])
    sc = np.asarray(new[s]['pred_scores']).ravel()
    for i in idx:
        m = np.asarray(new[s]['pred_masks'])[:, i] > 1e-3
        h = float(np.ptp(pts[m][:, 2]))
        hmin = min(hmin, h)
        top.append((sc[i], s, d[i], h))
top.sort(reverse=True)
print(f"  flips (all scores): {tot}   shortest leaf: {hmin:.2f} m")
for sc, s, d, h in top[:3]:
    print(f"    {s}  score {sc:.2f}  {d:.1f} deg  h={h:.2f} m")
PYEOF

echo "[3/4] integrity"
RAW="$RAW" OUT="$OUT" $PY - <<'PYEOF'
import os, pickle, json, numpy as np
new = pickle.load(open(os.environ['OUT'], 'rb'))
sp = json.load(open('/data/challenge/data_download/articulate3d_splits.json'))
assert set(new) == set(sp['test']), "scene set mismatch"
n = np.concatenate([np.linalg.norm(np.asarray(v['pred_axises']), axis=1) for v in new.values()])
assert abs(n.min() - 1) < 1e-4 and abs(n.max() - 1) < 1e-4, "axes not unit"
c = np.concatenate([np.asarray(v['pred_classes']).ravel() for v in new.values()])
assert set(c.tolist()) <= {1, 2}, "unexpected class ids"
del c, n
raw = pickle.load(open(os.environ['RAW'], 'rb'))
for s in new:
    assert np.array_equal(np.asarray(raw[s]['pred_masks']),
                          np.asarray(new[s]['pred_masks'])), f"masks changed in {s}"
print(f"  OK: {len(new)} scenes, masks bit-identical, axes unit, classes in {{1,2}}")
PYEOF

echo "[4/4] packaging"
rm -rf "$SUB" && mkdir -p "$SUB"      # transfer_preds_files.py appends; must be clean
$PY transfer_preds_files.py --preds_pkl_dir "$OUT" --preds_dir "$SUB/mov_test" 2>&1 | tail -1
cd "$SUB" && zip -rq mov_test.zip mov_test
echo "READY: $SUB/mov_test.zip  ($(ls mov_test | wc -l) scene files)"
md5sum mov_test.zip
