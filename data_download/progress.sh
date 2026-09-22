#!/bin/bash
# One-shot progress report for the ScanNet++ download.
set -u
ROOT=/data/challenge/scannetpp_data
HERE="$(cd "$(dirname "$0")" && pwd)"
TOTAL_SCENES=1018

done_scenes=$(ls "$ROOT/data" 2>/dev/null | wc -l)
size=$(du -sh "$ROOT" 2>/dev/null | cut -f1)
workers=$(pgrep -fc "run_download.py" || true)
echo "$(date -Is)  size=$size  scenes_started=$done_scenes/$TOTAL_SCENES  workers=$workers"
if [ -f "$HERE/logs/status.log" ]; then
    echo "--- status ---"; cat "$HERE/logs/status.log"
fi
echo "--- recent hard failures ---"
grep -h -E "GAVE UP|gave up|SystemExit|expired|401|406" "$HERE"/logs/worker_*.log 2>/dev/null | tail -5
echo "--- restarts per worker ---"
for f in "$HERE"/logs/worker_*.log; do
    [ -e "$f" ] || continue
    printf "%s: %s starts\n" "$(basename "$f" .log)" "$(grep -c '=== start attempt' "$f")"
done
