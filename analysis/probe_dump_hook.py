#!/usr/bin/env python
"""Dump the exact 256-d vector the axis head consumes, per matched instance.

Decides whether a LoRA/DoRA-adapted feature path could possibly fix lateral
hinges: if a linear probe on these frozen features cannot separate
lateral-axis from vertical-axis rotations, no amount of head capacity or
adapter tuning downstream of them will either -- the information is not there.

Monkey-patches Mask3D._axis_bin_residual to record its `feat` argument
(= cat[queries, mask-pooled point features]) alongside the scene id, then
runs the standard val inference path.
"""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "/data/challenge/USDNet")

OUT = "/data/challenge/analysis/probe_feats.pkl"
store = {"feat": [], "scene": []}


def install(model, scene_getter):
    import models.mask3d as M
    orig = M.Mask3D._axis_bin_residual

    def patched(self, feat, axis_free):
        store["feat"].append(feat.detach().float().cpu().numpy()[0])   # (Q, 256)
        store["scene"].append(scene_getter())
        return orig(self, feat, axis_free)

    M.Mask3D._axis_bin_residual = patched
