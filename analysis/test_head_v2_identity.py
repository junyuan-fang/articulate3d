#!/usr/bin/env python
"""Prove the articulation head v2 is a strict, identity-initialised extension.

Two claims are checked on the real module (not a reimplementation):

1. origin_agg="attn" with its zero-initialised gate produces bit-comparable
   origins to origin_agg="mean". A masked softmax of all-zero logits is 1/N on
   every masked point, which is the uniform mean the stock head takes -- so a
   checkpoint loaded into the new model cannot silently move AP.

2. axis_bin_residual=True at init returns, for every query, whichever of
   {gravity, the free axis flattened into the ground plane} is closer to the
   free axis. That is the "vert-or-horiz hard" rule, the best of the
   training-free axis rules at AP50_axis 0.4677 vs 0.4432 unsnapped
   (analysis/pp_sweep3.py).

Run: PYTHONPATH=/data/challenge/USDNet python test_head_v2_identity.py
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, N, Q, D, A = 2, 5000, 32, 128, 64


class Head(torch.nn.Module):
    """Same layers and same maths as Mask3D's mode-2 origin branch."""

    def __init__(self):
        super().__init__()
        self.origin_attn_dim = A
        self.origin_attn_temp = 1.0
        self.origin_attn_key = torch.nn.Sequential(
            torch.nn.Linear(D + 3, D), torch.nn.ReLU(), torch.nn.Linear(D, A))
        self.origin_attn_query = torch.nn.Sequential(
            torch.nn.Linear(D, D), torch.nn.ReLU(), torch.nn.Linear(D, A))
        self.origin_attn_bias = torch.nn.Linear(D + 3, 1)
        self.origin_attn_out = torch.nn.Linear(A, A, bias=False)
        torch.nn.init.zeros_(self.origin_attn_out.weight)
        torch.nn.init.zeros_(self.origin_attn_bias.weight)
        torch.nn.init.zeros_(self.origin_attn_bias.bias)

    _origin_attention = None  # bound below


import models.mask3d as m3d  # noqa: E402
Head._origin_attention = m3d.Mask3D._origin_attention


def main():
    feats = torch.randn(B, N, D, device=DEV)
    coords = torch.randn(B, N, 3, device=DEV) * 3
    queries = torch.randn(B, Q, D, device=DEV)
    per_point = torch.randn(B, N, 3, device=DEV) + coords
    masks = torch.rand(B, N, Q, device=DEV) > 0.7
    masks[:, :, 0] = False                      # an empty query must not NaN
    mask_f = masks.float()
    n_pts = mask_f.sum(1, keepdim=True).permute(0, 2, 1)   # (B, Q, 1)

    head = Head().to(DEV)
    w = head._origin_attention(feats, coords, queries, masks)
    attn_out = torch.einsum("bnq,bnd->bqd", w, per_point)
    mean_out = torch.einsum("bnq,bnd->bqd", mask_f, per_point) / (n_pts + 1e-6)

    ok = masks.any(1)                            # (B, Q) non-empty queries
    diff = (attn_out - mean_out)[ok].abs().max().item()
    print(f"[origin] max |attn - mean| over non-empty queries: {diff:.3e}")
    assert torch.isfinite(attn_out).all(), "empty query produced NaN"
    assert diff < 1e-4, "attention at init is NOT the uniform mean"

    # gradient must reach the new parameters even though they start at zero
    attn_out.sum().backward()
    gnorm = head.origin_attn_bias.weight.grad.abs().sum().item()
    print(f"[origin] grad reaching origin_attn_bias: {gnorm:.3e}")
    assert gnorm > 0, "zero-init gate blocks all gradient -- head cannot train"

    # ---- axis bin+residual ------------------------------------------------
    class AxHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.axis_prior_temp = 10.0
            self.axis_res_scale = 0.0
            self.axis_cls_head = torch.nn.Sequential(
                torch.nn.Linear(2 * D, D), torch.nn.ReLU(), torch.nn.Linear(D, 2))
            self.axis_res_head = torch.nn.Sequential(
                torch.nn.Linear(2 * D, D), torch.nn.ReLU(), torch.nn.Linear(D, 6))
            for h in (self.axis_cls_head, self.axis_res_head):
                torch.nn.init.zeros_(h[-1].weight)
                torch.nn.init.zeros_(h[-1].bias)
    AxHead._axis_bin_residual = m3d.Mask3D._axis_bin_residual

    ax = AxHead().to(DEV)
    feat = torch.randn(B, Q, 2 * D, device=DEV)
    free = torch.randn(B, Q, 3, device=DEV)
    out = F.normalize(ax._axis_bin_residual(feat, free), dim=-1)

    # replay the reference rule the head is supposed to reproduce at init
    f = F.normalize(free, dim=-1)
    up = torch.tensor([0., 0., 1.], device=DEV)
    flat = F.normalize(f - f[..., 2:3] * up, dim=-1)
    want = torch.where((f[..., 2].abs() >= 2 ** -0.5)[..., None],
                       up.expand_as(f), flat)
    err = torch.rad2deg(torch.acos(
        torch.einsum("bqd,bqd->bq", out, want).abs().clamp(max=1.0)))
    print(f"[axis] max deviation from the vert-or-horiz rule: {err.max():.4f} deg")
    # acos is ill-conditioned at 1: an fp32 cosine error of 6e-8 already shows up
    # as 0.02 deg, so the tolerance is on angle, not on cosine
    assert err.max() < 0.05, "warm start does not reproduce the vert-or-horiz rule"

    ang = torch.rad2deg(torch.acos(
        torch.einsum("bqd,bqd->bq", out, f).abs().clamp(max=1.0)))
    print(f"[axis] angle to the free axis it replaces: "
          f"median {ang.median():.1f} deg, max {ang.max():.1f} deg")

    out.sum().backward()
    g = ax.axis_cls_head[-1].weight.grad.abs().sum().item()
    print(f"[axis] straight-through grad reaching the classifier: {g:.3e}")
    assert g > 0, "argmax blocks the gradient -- classifier cannot train"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
