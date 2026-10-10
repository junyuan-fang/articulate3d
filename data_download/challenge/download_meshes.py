#!/usr/bin/env python
"""Download the aligned meshes the Articulate3D challenge needs.

Same job as the challenge page's download_meshes.py, with three changes that
matter given how slow and flaky the TUM server is from here:

  - parallel: the official script is serial at ~0.3 MB/s per connection
  - retries timeouts/429/5xx instead of dying on the first one
  - a 404 records the scene as unavailable and moves on, rather than aborting
    the whole run (scene 280b83fcf3 is in Articulate3D's train split but no
    longer exists in the current ScanNet++ release)

Meshes land in the same data_root as the full download, at the identical
path, so the full download later skips whatever this already fetched.
"""
import argparse
import getpass
import re
from urllib.parse import quote
import io
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
ROOT_URL = ("https://scannetpp.mlsg.cit.tum.de/scannetpp/download/v2"
            "?version=v1&token=TOKEN&file=FILEPATH")
REMOTE = "data/{scene}/scans/mesh_aligned_0.05.zip"
LOCAL_PLY = "data/{scene}/scans/mesh_aligned_0.05.ply"
MAX_TRIALS = 8

socket.setdefaulttimeout(120)
_print_lock = threading.Lock()
_active_token = ""


def redact(value):
    text = str(value)
    if _active_token:
        text = text.replace(_active_token, "[REDACTED]")
        text = text.replace(quote(_active_token, safe=""), "[REDACTED]")
    return re.sub(r"(?i)(token=)[^&\s\"'<>]+", r"\1[REDACTED]", text)


def read_token():
    token = os.environ.get("SCANNETPP_TOKEN", "").strip()
    if not token and sys.stdin.isatty():
        token = getpass.getpass("ScanNet++ download token (hidden): ").strip()
    if not token or token == "<YOUR_TOKEN_HERE>":
        raise SystemExit("Set SCANNETPP_TOKEN before unattended downloads.")
    return token


def log(msg):
    with _print_lock:
        print(redact(msg), flush=True)


def fetch(url):
    """Return response bytes, or None if the server says 404."""
    for attempt in range(MAX_TRIALS):
        try:
            with urllib.request.urlopen(url) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 401:
                log("FATAL: 401 unauthorized -- token invalid or expired")
                os._exit(2)
            if e.code != 429 and not (500 <= e.code < 600):
                raise
            err = e
        except (urllib.error.URLError, OSError) as e:
            err = e
        delay = min(120, 5 * 2 ** attempt)
        log(f"  retry {attempt + 1}/{MAX_TRIALS} in {delay}s ({err})")
        time.sleep(delay)
    raise RuntimeError(f"gave up: {redact(url)}") from None


def get_scene(scene, token, data_root):
    ply = data_root / LOCAL_PLY.format(scene=scene)
    if ply.is_file():
        return "skip"
    url = ROOT_URL.replace("TOKEN", quote(token, safe="")).replace("FILEPATH", REMOTE.format(scene=scene))
    blob = fetch(url)
    if blob is None:
        log(f"MISSING (404): {scene}")
        return "missing"
    ply.parent.mkdir(parents=True, exist_ok=True)
    # Extract from memory so an interrupted run can never leave a truncated
    # .ply that a later run would mistake for a finished download.
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(ply.parent)
    if not ply.is_file():
        log(f"WARN: {scene} zip did not contain {ply.name}")
        return "bad"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data/challenge/scannetpp_data")
    ap.add_argument("--splits", default=str(HERE.parent / "articulate3d_splits.json"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    global _active_token
    _active_token = read_token()
    data_root = Path(args.data_root)
    splits = json.loads(Path(args.splits).read_text())
    scenes = splits["train"] + splits["val"] + splits["test"]
    log(f"{len(scenes)} scenes -> {data_root}")

    counts, done = {}, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(get_scene, s, _active_token, data_root): s for s in scenes}
        for f in futs:
            pass
        for f, scene in futs.items():
            try:
                r = f.result()
            except Exception as e:
                log(f"ERROR {scene}: {e}")
                r = "error"
            counts[r] = counts.get(r, 0) + 1
            done += 1
            if done % 10 == 0:
                log(f"[{done}/{len(scenes)}] {counts}")

    log(f"FINAL {counts}")
    missing = [s for s in scenes
               if not (data_root / LOCAL_PLY.format(scene=s)).is_file()]
    (HERE / "missing_scenes.txt").write_text("\n".join(missing) + "\n" if missing else "")
    log(f"{len(scenes) - len(missing)}/{len(scenes)} meshes present; missing: {missing}")
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
