#!/bin/bash
# Pull the Articulate3D annotations, retrying until complete.
#
# snapshot_download's own retries aren't enough on this link -- it died with an
# httpx.ReadTimeout partway through. Already-downloaded files are skipped on
# re-entry, so looping just resumes.
set -u
PY=/data/challenge/data_download/.venv/bin/python
OUT=/data/challenge/articulate3d_annotations
EXPECTED=477

for i in $(seq 1 60); do
    n=$(find "$OUT" -name "*.json" 2>/dev/null | wc -l)
    if [ "$n" -ge "$EXPECTED" ]; then
        echo "complete: $n/$EXPECTED json files"
        exit 0
    fi
    echo "=== attempt $i ($n/$EXPECTED present) at $(date -Is) ==="
    $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('INSAIT-Institute/Articulate3D', repo_type='dataset',
                  local_dir='$OUT', max_workers=4)
" && break
    sleep 15
done

n=$(find "$OUT" -name "*.json" 2>/dev/null | wc -l)
echo "final: $n/$EXPECTED json files"
