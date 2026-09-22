#!/usr/bin/env python
"""Robust runner for the official ScanNet++ download script.

Wraps download_scannetpp.py without modifying it. Three fixes, all needed
because the TUM server is slow and flaky from here and this job runs for days:

  - socket timeout: the official script sets none, so a stalled connection
    hangs forever instead of failing and retrying
  - URLError/OSError retry: urlretrieve_multi_trials only catches HTTPError
    and ContentTooShortError, so a connection timeout kills the whole run
  - .part + rename: urlretrieve writes straight to the target path, so a
    killed process leaves a truncated file that the resume check ("file
    exists, skip") then treats as complete

Usage: python run_download.py <config.yml>
"""
import builtins
import os
import socket
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
socket.setdefaulttimeout(120)

import download_scannetpp as ds

# The confirmation prompt would block on restart; we are explicitly running
# an unattended download, so answer it.
builtins.input = lambda *a, **k: "y"

_orig_retrieve = ds.urlretrieve_multi_trials
OUTER_TRIALS = 10


def _retrieve(url, filename, max_trials=5):
    # pid in the name so two workers can never share a scratch file
    part = Path(f"{filename}.{os.getpid()}.part")
    for attempt in range(OUTER_TRIALS):
        try:
            ok = _orig_retrieve(url, part, max_trials=max_trials)
            if ok:
                os.replace(part, filename)
            return ok
        except urllib.error.HTTPError as e:
            # 401/404/406 exit inside _orig_retrieve; what reaches here is a
            # 429/5xx that already burned its inner retries, plus anything odd.
            if e.code != 429 and not (500 <= e.code < 600):
                raise
            err = e
        except (urllib.error.URLError, OSError) as e:
            err = e
        if part.exists():
            part.unlink()
        delay = min(120, 5 * 2 ** attempt)
        print(f"WARN: {ds.redact(err)} on {ds.redact(url)} -- outer retry {attempt + 1}/{OUTER_TRIALS} in {delay}s",
              flush=True)
        time.sleep(delay)
    raise RuntimeError(f"gave up after {OUTER_TRIALS} outer retries: {ds.redact(url)}")


ds.urlretrieve_multi_trials = _retrieve


class _Args:
    def __init__(self, config_file):
        self.config_file = config_file


if __name__ == "__main__":
    ds.run(_Args(sys.argv[1]))
