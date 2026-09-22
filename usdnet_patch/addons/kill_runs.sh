#!/bin/bash
# Kill any running training/inference for this repo.
#
# Lives in a file for a reason: `pkill -f main_instance_segmentation` also
# matches the command line of the shell running it, so an inline version kills
# its own caller (exit 144). Same trap as data_download/stop.sh.
#
# The pattern must match the *relative* invocation the scripts actually use
# (they `cd` to the repo root first, so argv[1] is just the bare filename); an
# earlier version keyed on the absolute path and silently matched nothing.
# Restricting to processes whose comm is `python` keeps this shell and the
# wrapper bash out of the kill list.
set -u
SELF=$$
PAT="(main_instance_segmentation_articulation|eval_val)\.py"

victims() {
    for pid in $(pgrep -f "$PAT" 2>/dev/null); do
        [ "$pid" = "$SELF" ] && continue
        # the main process reports comm=python; PL's dataloader workers rename
        # themselves to pt_data_worker and are orphaned (reparented to init) when
        # the parent dies -- they keep the GPU allocation alive, so kill them too
        case "$(cat /proc/$pid/comm 2>/dev/null)" in
            python*|pt_data_worker*) echo "$pid";;
        esac
    done
}

for pid in $(victims); do kill "$pid" 2>/dev/null; done
sleep 5
for pid in $(victims); do kill -9 "$pid" 2>/dev/null; done
sleep 3
echo "remaining: $(victims | wc -l)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
