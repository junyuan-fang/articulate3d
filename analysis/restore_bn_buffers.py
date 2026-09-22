"""Copy the source checkpoint's BatchNorm running stats back into a finetuned ckpt.

`requires_grad_(False)` freezes gradients but not BN buffers, so a run that was
supposed to touch only the origin branch also drifted 186 running_mean /
running_var buffers towards 4 m-crop statistics (analysis/check_ckpt_diff.py).
Restoring them makes the backbone, the decoder, pred_masks and pred_axises
bit-identical to the source again, which is what the ablation assumes.

usage: restore_bn_buffers.py <src.ckpt> <finetuned.ckpt> <out.ckpt>
"""
import sys
import torch

src_p, ft_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
src = torch.load(src_p, map_location="cpu", weights_only=False)
ft = torch.load(ft_p, map_location="cpu", weights_only=False)

BUF = ("running_mean", "running_var", "num_batches_tracked")
n = 0
for k, v in src["state_dict"].items():
    if k.split(".")[-1] in BUF and k in ft["state_dict"]:
        if not torch.equal(ft["state_dict"][k], v):
            n += 1
        ft["state_dict"][k] = v.clone()
torch.save(ft, out_p)
print(f"restored {n} drifted BN buffers from {src_p}\nwrote {out_p}")
