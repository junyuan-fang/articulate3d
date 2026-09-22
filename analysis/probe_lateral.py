#!/usr/bin/env python
"""Linear probe: do the frozen features encode "this hinge is lateral"?

Self-contained: loads the checkpoint, runs the val scenes through the model
with a hook on the axis-head input, matches queries to GT, then fits a
logistic probe with leave-one-scene-out CV (only ~7 positives exist, so
scene-level LOO is the only honest protocol).

Verdict:
  AUC ~ 0.5  -> information absent; LoRA/DoRA on the feature path cannot help
  AUC > 0.7  -> information present; the head is the bottleneck -> adapters
                are worth the engineering
"""
import os
import pickle
import sys

import h5py
import numpy as np

sys.path.insert(0, "/data/challenge/USDNet")
sys.path.insert(0, "/data/challenge/analysis")

FEATS = "/data/challenge/analysis/probe_feats_matched.pkl"
VAL = "/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/validation"

rows = pickle.load(open(FEATS, "rb"))          # list of (scene, feat256, is_lateral)
X = np.stack([r[1] for r in rows]).astype(np.float64)
y = np.array([r[2] for r in rows], dtype=int)
scenes = np.array([r[0] for r in rows])

print(f"instances {len(y)}  lateral {y.sum()}  vertical {(1-y).sum()}  dim {X.shape[1]}")
if y.sum() < 3:
    print("too few positives to probe")
    sys.exit(0)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# leave-one-scene-out
probs = np.zeros(len(y))
for s in np.unique(scenes):
    te = scenes == s
    tr = ~te
    if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
        probs[te] = y[tr].mean()
        continue
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
    clf.fit(sc.transform(X[tr]), y[tr])
    probs[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

auc = roc_auc_score(y, probs)
print(f"leave-one-scene-out AUC = {auc:.3f}")
# permutation null
rng = np.random.RandomState(0)
null = [roc_auc_score(rng.permutation(y), probs) for _ in range(2000)]
p = float(np.mean(np.array(null) >= auc))
print(f"permutation p-value = {p:.4f}   (null AUC mean {np.mean(null):.3f})")
print(f"lateral mean prob {probs[y==1].mean():.3f} vs vertical {probs[y==0].mean():.3f}")
