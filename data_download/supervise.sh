#!/bin/bash
# Run all shard workers in parallel, restarting any that die.
#
# The download is resume-safe (existing files are skipped), so restarting a
# crashed worker just picks up where it left off. Workers keep restarting
# until their shard completes cleanly (exit 0) or MAX_RESTARTS is hit -- the
# cap stops a permanently-failing asset from spinning forever.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HERE/logs"
MAX_RESTARTS=${MAX_RESTARTS:-500}
mkdir -p "$LOGDIR"

# Scratch files left by a previous run's interrupted transfers. Real files are
# only renamed into place on success, so these are always safe to drop.
find /data/challenge/scannetpp_data -name "*.part" -delete 2>/dev/null

run_worker() {
    local cfg="$1" name log n=0
    name="$(basename "$cfg" .yml)"
    log="$LOGDIR/$name.log"
    while :; do
        echo "=== start attempt $((n + 1)) at $(date -Is) ===" >>"$log"
        "$HERE/.venv/bin/python" "$HERE/run_download.py" "$cfg" >>"$log" 2>&1
        local rc=$?
        echo "=== exit $rc at $(date -Is) ===" >>"$log"
        if [ $rc -eq 0 ]; then
            echo "$name: done" >>"$LOGDIR/status.log"
            return 0
        fi
        n=$((n + 1))
        if [ $n -ge $MAX_RESTARTS ]; then
            echo "$name: GAVE UP after $n restarts (last rc=$rc)" >>"$LOGDIR/status.log"
            return 1
        fi
        sleep 20
    done
}

pids=()
for cfg in "$HERE"/shards/worker_*.yml; do
    run_worker "$cfg" &
    pids+=($!)
    sleep 3   # stagger so all workers don't hit the server in the same second
done
echo "started ${#pids[@]} workers at $(date -Is)" >>"$LOGDIR/status.log"
wait "${pids[@]}"
echo "all workers finished at $(date -Is)" >>"$LOGDIR/status.log"
