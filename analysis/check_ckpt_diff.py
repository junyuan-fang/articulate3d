"""Which tensors actually moved during finetuning?

Decisive test for `general.train_origin_only`: if gradients really only reached
the origin branch, every other tensor must be bit-identical to the source
checkpoint, and therefore `pred_masks` and `pred_axises` are bit-identical too
(=> AP50 and AP50_axis cannot move).
"""
import sys
import torch

src = sys.argv[1] if len(sys.argv) > 1 else "/data/challenge/ckpts/checkpoints/mov_trainval.ckpt"
dst = sys.argv[2]

a = torch.load(src, map_location="cpu", weights_only=False)["state_dict"]
b = torch.load(dst, map_location="cpu", weights_only=False)["state_dict"]

only_a = sorted(set(a) - set(b))
only_b = sorted(set(b) - set(a))
changed, same = [], []
for k in sorted(set(a) & set(b)):
    ta, tb = a[k], b[k]
    if ta.shape != tb.shape:
        changed.append((k, float("nan")))
        continue
    d = (ta.float() - tb.float()).abs().max().item()
    (changed if d > 0 else same).append((k, d))

print(f"src {src}\ndst {dst}\n")
print(f"tensors: shared {len(same)+len(changed)}  only-in-src {len(only_a)}  only-in-dst {len(only_b)}")
print(f"BIT-IDENTICAL: {len(same)}   CHANGED: {len(changed)}\n")
print("CHANGED tensors (max abs delta):")
for k, d in changed:
    print(f"  {d:12.6g}  {k}")
print("\nonly in dst (new v2 params):")
for k in only_b:
    print(f"  {tuple(b[k].shape)}  {k}   |w|max={b[k].float().abs().max().item():.6g}")
if only_a:
    print("\nonly in src:")
    for k in only_a:
        print(f"  {k}")

# Anything outside the origin branch that moved is a leak.
leak = [k for k, _ in changed
        if not any(t in k for t in ("origin_embed_head", "origin_attn_"))]
print("\n" + ("LEAK: non-origin tensors changed -> " + str(leak) if leak
              else "OK: every changed tensor is in the origin branch."))
