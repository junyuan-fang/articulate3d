#!/usr/bin/env python
"""Build an x-mirrored copy of the validation split for flip-TTA.

Row order is untouched, so predicted masks align row-for-row with the ORIGINAL
point cloud; only geometry needs mirroring back afterwards. Per-scene mirror
constant C (x' = C - x) is recorded for the inverse transform.

The h5/instance files are copied untouched: eval metrics printed by an
inference run on this data are meaningless (GT axes are in the unflipped
frame) -- the run exists solely to harvest preds.pkl.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

SRC = Path("/data/challenge/USDNet/data/processed/articulate3d_challenge_mov")
DST = Path("/data/challenge/USDNet/data/processed/articulate3d_challenge_mov_flipx")

(DST / "validation").mkdir(parents=True, exist_ok=True)
consts = {}
for f in sorted((SRC / "validation").glob("*.npy")):
    a = np.load(f).copy()
    C = float(a[:, 0].max() + a[:, 0].min())
    a[:, 0] = C - a[:, 0]          # mirror x
    a[:, 6] = -a[:, 6]             # normal x component
    np.save(DST / "validation" / f.name, a)
    consts[f.stem] = C
for f in sorted((SRC / "validation").glob("*_articulation.h5")):
    shutil.copy(f, DST / "validation" / f.name)

# database with filepaths pointing at the mirrored npys
db = yaml.safe_load(open(SRC / "validation_database.yaml"))
for e in db:
    e["filepath"] = e["filepath"].replace(
        "articulate3d_challenge_mov/validation",
        "articulate3d_challenge_mov_flipx/validation")
yaml.safe_dump(db, open(DST / "validation_database.yaml", "w"), sort_keys=False)

json.dump(consts, open(DST / "flip_constants.json", "w"))
print(f"{len(consts)} scenes mirrored -> {DST}")
