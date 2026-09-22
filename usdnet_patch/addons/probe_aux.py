"""Per-FPN-level probe: is the lateral-hinge signal present anywhere in the backbone?

The earlier probe tested only the axis head's input (the final 256-d vector) and
found nothing (AUC 0.247, p=0.99). That rules out adapters *downstream* of it,
but not the possibility that the information lives in a shallower FPN level and
is destroyed by aggregation. Ladder Side-Tuning could recover exactly that -- so
this decides whether an LST side network has anything to work with.

For every matched instance we mask-pool each aux[l] over the part's points and
bank the pooled vector per level.
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
OUT = "/data/challenge/analysis/probe_aux_feats.pkl"

_cap = {"aux": None}
_orig_fwd = M.Mask3D.forward


def _patched_forward(self, x, point2segment=None, raw_coordinates=None, is_eval=False, **kw):
    out = _orig_fwd(self, x, point2segment=point2segment,
                    raw_coordinates=raw_coordinates, is_eval=is_eval, **kw)
    return out


def _hook_backbone(model):
    """Capture aux by wrapping the backbone call itself."""
    bb = model.model.backbone
    orig = bb.forward

    def fwd(x):
        res = orig(x)
        if isinstance(res, tuple):
            _cap["aux"] = res[1]
        return res

    bb.forward = fwd


_rows = []


def _wrap(model):
    orig_step = model.eval_step

    def step(batch, batch_idx):
        out = orig_step(batch, batch_idx)
        try:
            _, _, file_names = batch
            scene = file_names[0]
            aux = _cap["aux"]
            preds = model.preds.get(scene)
            if aux is None or preds is None:
                return out
            # decomposed per-level point features for batch item 0
            levels = []
            for lv in aux:
                f = lv.decomposed_features[0].detach().float().cpu().numpy()
                levels.append(f)
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
                    pooled = []
                    for f in levels:
                        # aux levels are coarser than the full cloud; subsample
                        # the mask proportionally to pool over the same region
                        if len(f) == len(m):
                            sel = m
                        else:
                            idx = (np.where(m)[0] * (len(f) / len(m))).astype(int)
                            idx = np.clip(idx, 0, len(f) - 1)
                            sel = np.zeros(len(f), bool)
                            sel[np.unique(idx)] = True
                        pooled.append(f[sel].mean(0) if sel.any() else np.zeros(f.shape[1]))
                    _rows.append((scene, [p.astype(np.float32) for p in pooled], gt[bid]))
            print(f"[aux] {scene}: total {len(_rows)}", flush=True)
        except Exception as e:
            print("[aux] skip:", repr(e)[:120], flush=True)
        finally:
            _cap["aux"] = None
        return out

    model.eval_step = step


@hydra.main(config_path="conf", config_name="config_articulation_instance_segmentation.yaml")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    _hook_backbone(model)
    _wrap(model)
    runner = Trainer(gpus=cfg.general.gpus, logger=loggers,
                     weights_save_path=str(cfg.general.save_dir), **cfg.trainer)
    runner.validate(model)
    pickle.dump(_rows, open(OUT, "wb"))
    print("[aux] wrote", OUT, len(_rows))


if __name__ == "__main__":
    main()
