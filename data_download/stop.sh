#!/bin/bash
# Stop the supervisor and all workers.
#
# Lives in a file rather than being typed inline: `pkill -f run_download.py`
# also matches the command line of the shell running it, so an inline pkill
# kills its own caller.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

# Supervisor first, or it just restarts the workers we are about to kill.
# Match on the basename: the supervisor is usually launched as ./supervise.sh,
# so an absolute-path pattern silently misses it -- which once left a stale
# supervisor respawning workers for hours after a "successful" stop.
pkill -f "supervise\.sh" 2>/dev/null
sleep 2
pkill -9 -f "supervise\.sh" 2>/dev/null
sleep 1
pkill -f "$HERE/.venv/bin/python $HERE/run_download.py" 2>/dev/null
sleep 3
pkill -9 -f "$HERE/.venv/bin/python $HERE/run_download.py" 2>/dev/null

left=$(pgrep -fc "$HERE/run_download.py" 2>/dev/null || echo 0)
echo "workers still running: $left"

# Scratch files from interrupted transfers; the real files were renamed into
# place only on success, so these are always safe to drop.
n=$(find /data/challenge/scannetpp_data -name "*.part" -print -delete 2>/dev/null | wc -l)
echo "removed $n partial files"
