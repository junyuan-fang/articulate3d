"""Dump the axis head's input vector per matched instance (LoRA/DoRA feasibility probe).

Uses the repo's own @hydra.main entry (hydra 1.0.5 has no compose API) and the
same code path as eval_val.py, with a monkey-patch on _axis_bin_residual that
records its `feat` argument -- exactly what an adapter would have to improve.
"""
import os
import pickle

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

import models.mask3d as M
from main_instance_segmentation_articulation import get_parameters

VAL = "data/processed/articulate3d_challenge_mov/validation"
OUT = "/data/challenge/analysis/probe_feats_matched.pkl"

_cap = {"feat": None}
_orig = M.Mask3D._axis_bin_residual


def _patched(self, feat, axis_free):
    _cap["feat"] = feat.detach().float().cpu().numpy()[0]
    return _orig(self, feat, axis_free)


M.Mask3D._axis_bin_residual = _patched
_rows = []
_orig_eval_step = None


def _wrap_trainer(model):
    """After each val step, match that scene's queries to GT and bank features."""
    orig_step = model.eval_step

    def step(batch, batch_idx):
        out = orig_step(batch, batch_idx)
        try:
            _, _, file_names = batch
            scene = file_names[0]
            feats = _cap["feat"]
            preds = model.preds.get(scene)
            if feats is None or preds is None:
                return out
            a = np.load(os.path.join(VAL, scene + ".npy"), mmap_mode="r")
            inst = np.asarray(a[:, 10]).astype(int)
            gt = {}
            with h5py.File(os.path.join(VAL, scene + "_articulation.h5"), "r") as h:
                for k in h.keys():
                    g = h[k]
                    if int(np.asarray(g["sem_id"])) == 1:
                        ax = np.asarray(g["axis"], float)
                        gt[int(k)] = int(abs(ax[2] / max(np.linalg.norm(ax), 1e-9)) < 0.5)
            if not gt:
                return out
            pm = np.asarray(preds["pred_masks"]) > 1e-3
            if pm.shape[0] != len(inst):
                return out
            sc = np.asarray(preds["pred_scores"]).ravel()
            gmasks = {i: inst == i for i in gt}
            used = set()
            for qi in np.argsort(-sc):
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
                    if qi < feats.shape[0]:
                        _rows.append((scene, feats[qi].copy(), gt[bid]))
            print(f"[probe] {scene}: total {len(_rows)}", flush=True)
        except Exception as e:
            print("[probe] skip:", e, flush=True)
        finally:
            _cap["feat"] = None
        return out

    model.eval_step = step


@hydra.main(config_path="conf", config_name="config_articulation_instance_segmentation.yaml")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    _wrap_trainer(model)
    runner = Trainer(gpus=cfg.general.gpus, logger=loggers,
                     weights_save_path=str(cfg.general.save_dir), **cfg.trainer)
    runner.validate(model)
    pickle.dump(_rows, open(OUT, "wb"))
    print("[probe] wrote", OUT, len(_rows))


if __name__ == "__main__":
    main()
