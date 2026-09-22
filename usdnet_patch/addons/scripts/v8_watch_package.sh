#!/bin/bash
# Night watch: as soon as v8's epoch-19 checkpoint lands, stop the run (the
# remaining 130 epochs are worthless per v3's measured curve), then build the
# final submission package from it. Morning = submit only.
set -u
cd "$(dirname "$0")/.."

until ls results/v8_tv150/epoch=19-*.ckpt >/dev/null 2>&1; do
    sleep 60
    pgrep -f "main_instance_segmentation_articulation" >/dev/null || {
        echo "TRAINING DIED before ep19"; exit 1; }
done
echo "ep19 checkpoint landed"
./kill_runs.sh

CK=$(ls results/v8_tv150/epoch=19-*.ckpt | head -1)
cp "$CK" results/v8_tv150/ep19.ckpt
CKPT=results/v8_tv150/ep19.ckpt bash scripts/endgame_package.sh
mv results/sub_v7 results/sub_v8 2>/dev/null || true
ls -la results/sub_v8/mov_test.zip 2>/dev/null || ls -la results/sub_v7/mov_test.zip
echo V8_PACKAGE_DONE
