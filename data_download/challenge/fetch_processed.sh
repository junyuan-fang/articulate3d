#!/bin/bash
# Fetch USDNet's preprocessed dataset (5.8 GB zip) from Google Drive.
#
# The plain `gdown --folder` run died at 1.76 GB with a broken connection and
# left a randomly-named .part it can't resume from. gdown -c resumes via HTTP
# Range against a deterministic "<output>.part", so we normalise the name and
# loop. The zip is integrity-checked at the end -- if a resume ever splices the
# byte stream wrong, that check catches it and we start over clean.
set -u
GDOWN=/data/challenge/data_download/.venv/bin/gdown
FILE_ID=1QS_D5CBoF5AssleA3kdMMirwFEWPMaMW
DEST_DIR=/data/challenge/USDNet/data/processed/Processed_dataset
OUT="$DEST_DIR/processed.zip"
PY=/data/challenge/data_download/.venv/bin/python

mkdir -p "$DEST_DIR"

# Adopt whatever partial data a previous run left behind.
if [ ! -f "$OUT.part" ]; then
    stray=$(find "$DEST_DIR" -maxdepth 1 -name "processed.zip*.part" | head -1)
    if [ -n "$stray" ]; then
        echo "adopting partial: $stray ($(du -h "$stray" | cut -f1))"
        mv "$stray" "$OUT.part"
    fi
fi

for i in $(seq 1 40); do
    if [ -f "$OUT" ]; then
        echo "=== verifying zip ==="
        if $PY -c "
import zipfile,sys
try:
    z=zipfile.ZipFile('$OUT')
    bad=z.testzip()
    sys.exit(1 if bad else 0)
except Exception as e:
    print(e); sys.exit(1)
"; then
            echo "zip OK: $(du -h "$OUT" | cut -f1)"
            exit 0
        fi
        echo "zip CORRUPT -- discarding and restarting from scratch"
        rm -f "$OUT" "$OUT.part"
    fi
    echo "=== attempt $i at $(date -Is) ($(du -ch "$OUT.part" 2>/dev/null | tail -1 | cut -f1) done) ==="
    $GDOWN --continue -O "$OUT" "$FILE_ID"
    sleep 10
done
echo "gave up after 40 attempts" >&2
exit 1
