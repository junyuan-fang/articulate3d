#!/usr/bin/env python
"""Prove the mode-2 aggregation rewrite in models/mask3d.py is exact.

The original code expands the per-point features to (B, N, Q, D) and multiplies
by the mask before summing. That materialises two ~120 MB tensors at 100k
points x 100 queries and keeps both alive for backward, which OOMs a 24 GB
card. The rewrite expresses the same reduction as mask^T @ per_point.

Comparing full inference runs cannot settle this: `general.seed` is null, so
masks, scores and classes differ between two runs of *identical* code. This
unit test is deterministic and isolates exactly the changed arithmetic.
"""
import torch

torch.manual_seed(0)
B, N, Q, D = 2, 5000, 100, 3
dev = "cuda" if torch.cuda.is_available() else "cpu"

coords = torch.randn(B, N, D, device=dev)
feat = torch.randn(B, N, D, device=dev)
mask = torch.rand(B, N, Q, device=dev) > 0.7
cnt = mask.unsqueeze(-1).sum(dim=1)                     # (B, Q, 1)

# --- original ---
co_e = coords.unsqueeze(2)                              # (B, N, 1, D)
m_e = mask.unsqueeze(-1)                                # (B, N, Q, 1)
f_e = feat.unsqueeze(2).expand(-1, -1, Q, -1)           # (B, N, Q, D)
old_origin = ((f_e + co_e) * m_e).sum(dim=1) / (cnt + 1e-6)
old_axis = (f_e * m_e).sum(dim=1) / (cnt + 1e-6)

# --- rewrite ---
mask_f = mask.to(feat.dtype)
new_origin = torch.einsum("bnq,bnd->bqd", mask_f, feat + coords) / (cnt + 1e-6)
new_axis = torch.einsum("bnq,bnd->bqd", mask_f, feat) / (cnt + 1e-6)

for name, a, b in [("origin", old_origin, new_origin), ("axis", old_axis, new_axis)]:
    d = (a - b).abs().max().item()
    ok = torch.allclose(a, b, rtol=1e-5, atol=1e-5)
    print(f"  {name:7s} max|diff| = {d:.3e}   allclose = {ok}")
    assert ok, f"{name} mismatch"

print("\nequivalent (differences are float32 summation-order noise only)")
