#!/usr/bin/env python
"""Per-FPN-level probe analysis: which layer, if any, knows about lateral hinges?

Same honest protocol as the final-feature probe: leave-one-scene-out logistic
regression with a permutation null, because only ~7 positives exist.

Reading the result:
  every level ~0.5      -> the information is nowhere in the backbone; Ladder
                           Side-Tuning has nothing to recover, drop the idea
  a shallow level >0.7  -> the signal exists early and is destroyed by
                           aggregation; an LST side network tapping that level
                           is well motivated
"""
import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROWS = "/data/challenge/analysis/probe_aux_feats.pkl"
rows = pickle.load(open(ROWS, "rb"))
if not rows:
    sys.exit("no rows")

scenes = np.array([r[0] for r in rows])
y = np.array([r[2] for r in rows], dtype=int)
n_lv = len(rows[0][1])
print(f"instances {len(y)}  lateral {y.sum()}  vertical {(1-y).sum()}  levels {n_lv}")
if y.sum() < 3:
    sys.exit("too few positives")

rng = np.random.RandomState(0)
for lv in range(n_lv):
    X = np.stack([r[1][lv] for r in rows]).astype(np.float64)
    probs = np.zeros(len(y))
    for s in np.unique(scenes):
        te = scenes == s
        tr = ~te
        if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
            probs[te] = y[tr].mean()
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        probs[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    auc = roc_auc_score(y, probs)
    null = [roc_auc_score(rng.permutation(y), probs) for _ in range(1000)]
    p = float(np.mean(np.array(null) >= auc))
    tag = "  <-- signal" if (auc > 0.7 and p < 0.05) else ""
    print(f"  aux[{lv}] dim={X.shape[1]:4d}  AUC={auc:.3f}  p={p:.3f}{tag}")
