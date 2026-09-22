#!/usr/bin/env python
"""Split the scene list into N per-worker configs.

The official script downloads strictly serially at ~0.3 MB/s per connection,
which would take ~2 months for the full dataset. Aggregate throughput scales
with concurrency (measured ~2.5 MB/s at 16 streams, with the server refusing
connections past ~12), so we run N copies of the script over disjoint scene
sets. Scenes are dealt round-robin so each shard gets a similar size mix.
"""
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
BASE_CFG = HERE / "download_scannetpp.yml"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

cfg = yaml.safe_load(BASE_CFG.read_text())
data_root = Path(cfg["data_root"])

# Union of every split, in split order, deduped.
scenes, seen = [], set()
for split in cfg["download_splits"]:
    for sid in (data_root / "splits" / f"{split}.txt").read_text().split():
        if sid not in seen:
            seen.add(sid)
            scenes.append(sid)

shard_dir = HERE / "shards"
shard_dir.mkdir(exist_ok=True)
for i in range(N):
    shard = scenes[i::N]
    out = dict(cfg)
    out.pop("token", None)
    out["download_scenes"] = shard
    out.pop("download_splits", None)
    (shard_dir / f"worker_{i:02d}.yml").write_text(yaml.safe_dump(out, sort_keys=False))

print(f"{len(scenes)} scenes -> {N} shards of ~{len(scenes) // N} in {shard_dir}")
