#!/usr/bin/env python
"""Run val inference with a hook that records the axis head's input vector,
then match queries to GT and save (scene, feat256, is_lateral) triples."""
import os
import pickle
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, "/data/challenge/USDNet")
os.chdir("/data/challenge/USDNet")

import models.mask3d as M
import hydra
from omegaconf import OmegaConf

VAL = "data/processed/articulate3d_challenge_mov/validation"
CKPT = "results/v3_origin_bnfix/ep19.ckpt"
OUT = "/data/challenge/analysis/probe_feats_matched.pkl"

# ---- hook ----
cap = {"feat": None}
_orig = M.Mask3D._axis_bin_residual


def patched(self, feat, axis_free):
    cap["feat"] = feat.detach().float().cpu().numpy()[0]
    return _orig(self, feat, axis_free)


M.Mask3D._axis_bin_residual = patched

# ---- build model exactly as infer_mov_val_v2 does ----
from hydra import initialize, compose
with initialize(config_path="conf"):
    cfg = compose(config_name="config_articulation_instance_segmentation",
                  overrides=[
                      "data/datasets=articulate3d_challenge_mov",
                      "general.num_targets=4",
                      "general.eval_on_segments=false",
                      "general.train_on_segments=false",
                      "general.eval_articulation=true",
                      "general.train_mode=false",
                      f"general.checkpoint={CKPT}",
                      "data.num_labels=3", "data.batch_size=1", "data.voxel_size=0.02",
                      "data.load_articulation=true", "data.use_hierarchy=false",
                      "data.cropping=false", "data.test_mode=validation",
                      "model.num_queries=100", "model.predict_articulation_mode=2",
                      "model.predict_articulation=true", "model.origin_agg=attn",
                      "model.axis_bin_residual=true", "model.axis_res_scale=0.1",
                      "loss.regular_arti_loss=false",
                  ])

from trainer.trainer import InstanceSegmentation
from utils.utils import load_checkpoint_with_missing_or_exsessive_keys

model = InstanceSegmentation(cfg)
cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)
model = model.cuda().eval()
model.prepare_data()
loader = model.test_dataloader()

rows = []
with torch.no_grad():
    for batch in loader:
        data, target, file_names = batch
        scene = file_names[0]
        import MinkowskiEngine as ME
        x = ME.SparseTensor(coordinates=data.coordinates, features=data.features, device="cuda")
        raw = data.raw_coordinates
        out = model.forward(x, point2segment=None, raw_coordinates=raw, is_eval=True)
        feats = cap["feat"]                                   # (Q, 256)
        if feats is None:
            continue
        masks = out["pred_masks"][0].detach().cpu().numpy()    # (N, Q)
        pm = (torch.sigmoid(torch.from_numpy(masks)).numpy() > 0.5)
        logits = out["pred_logits"][0].detach().cpu().numpy()
        scores = logits.max(-1)

        a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
        inst = np.asarray(a[:, 10]).astype(int)
        gt = {}
        with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
            for k in h.keys():
                g = h[k]
                if int(np.asarray(g["sem_id"])) == 1:
                    ax = np.asarray(g["axis"], float)
                    gt[int(k)] = abs(ax[2] / max(np.linalg.norm(ax), 1e-9)) < 0.5
        if not gt:
            continue
        gmasks = {i: inst == i for i in gt}
        # rows of pm correspond to voxelized points; align by length
        if pm.shape[0] != len(inst):
            continue
        used = set()
        for qi in np.argsort(-scores):
            m = pm[:, qi]
            if not m.any():
                continue
            best, bid = 0.0, None
            for gi, gm in gmasks.items():
                if gi in used:
                    continue
                iou = (m & gm).sum() / max((m | gm).sum(), 1)
                if iou > best:
                    best, bid = iou, gi
            if best > 0.5 and bid is not None:
                used.add(bid)
                rows.append((scene, feats[qi].copy(), int(gt[bid])))
        cap["feat"] = None
        print(f"{scene}: cumulative {len(rows)}", flush=True)

pickle.dump(rows, open(OUT, "wb"))
print("wrote", OUT, len(rows))
