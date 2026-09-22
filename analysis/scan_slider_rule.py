#!/usr/bin/env python
"""Is there a wide-margin separator between sliding leaves and drawer fronts?

feat_slider.py shows the nine genuine sliding leaves are tall panels (h
0.50-1.61 m) while the drawer-like group has median height 0.22 m. But eight of
the nine come from a single scene, so a threshold read off this val set is
worth about a third of its face value on test (docs/05-improvement.md). Only a
separator with a wide, obviously-physical margin is safe to deploy.

So: restrict to panels flat enough for the two-candidate question to even make
sense, then print every case sorted by height with its label, and look at where
the classes actually meet.
"""
import pickle

import numpy as np

rows = pickle.load(open("/tmp/slider_feats.pkl", "rb"))

for PL in (0.05, 0.06, 0.08):
    sub = [r for r in rows if r["pl"] < PL]
    pos = [r for r in sub if r["gi"] < r["gn"]]
    print(f"\n=== planarity < {PL}:  {len(sub)} parts, {len(pos)} slide in-plane ===")
    print(f"{'h':>6s} {'w':>6s} {'w/h':>6s} {'area':>6s} {'n':>6s}  label   scene")
    for r in sorted(sub, key=lambda x: -x["h"]):
        lab = "INPLANE" if r["gi"] < r["gn"] else "normal "
        print(f"{r['h']:6.2f} {r['w']:6.2f} {r['wh']:6.2f} {r['area']:6.2f} "
              f"{r['n']:6d}  {lab} {r['scene']}")
