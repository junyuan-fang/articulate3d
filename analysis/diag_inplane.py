#!/usr/bin/env python
"""Why does model arbitration fail on the ~89 deg cluster?

For every matched translation, print how far GT is from each of the two physical
candidates (panel normal, in-plane horizontal) and how far the MODEL is from
each -- the quantity arbitration actually decides on. If GT clearly prefers
in-plane but the model does not, the model is not a usable arbiter and the
discriminator has to be geometric instead.

Also reports the oracle over {model, normal, in-plane}: the ceiling any
selection rule can reach.
"""
import numpy as np

from test_two_candidate import ang, collect, flatten

rows = collect()
print(f"{'gt-nrm':>7s} {'gt-inp':>7s} {'mdl-nrm':>8s} {'mdl-inp':>8s} "
      f"{'plan':>6s} {'pick':>6s} {'best':>6s}")
n_amb = 0
for g, m, n_, ip_, pl in rows:
    if n_ is None:
        continue
    mh = flatten(m) if flatten(m) is not None else m
    gn, gi = ang(g, n_), ang(g, ip_)
    mn, mi = ang(mh, n_), ang(mh, ip_)
    if min(gn, gi) > 15:
        continue                       # neither candidate helps; not the issue here
    pick = "nrm" if mn < mi else "inp"
    best = "nrm" if gn < gi else "inp"
    if pick != best:
        n_amb += 1
        print(f"{gn:7.1f} {gi:7.1f} {mn:8.1f} {mi:8.1f} {pl:6.3f} "
              f"{pick:>6s} {best:>6s}   MISPICK")

print(f"\narbitration mispicks (where a candidate would have passed): {n_amb}")

orc = [min(ang(g, m), ang(g, n_) if n_ is not None else 9e9,
           ang(g, ip_) if ip_ is not None else 9e9) for g, m, n_, ip_, _ in rows]
orc = np.array(orc)
print(f"oracle over model/normal/in-plane: {np.mean(orc < 15)*100:.1f}% "
      f"({int((orc >= 15).sum())} unreachable)")

inp_wins = sum(1 for g, m, n_, ip_, pl in rows
               if n_ is not None and ang(g, ip_) < ang(g, n_))
print(f"GT prefers in-plane over normal: {inp_wins}/{len(rows)}")
