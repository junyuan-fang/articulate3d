#!/usr/bin/env python
"""Pull the per-validation articulation AP tables out of a training log.

The trainer prints one block per criterion (what_M / what_MA / what_MO /
what_MAO_ST) at every validation, with a rotation row, a translation row and an
average row. Reading them off the wandb offline file is painful, so parse the
text. AP_50% is the second number on each row.

Usage: python parse_train_log.py results/v2_full.log
"""
import re
import sys

ROW = re.compile(r"^(rotation|translation|average)\s*:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)")
# The mask-only table is printed with an empty tag ("what_"), so only the four
# articulation criteria are picked up here. AP50 is frozen by
# freeze_except_articulation anyway; confirm it once with analysis/score_pkl.py.
HEAD = re.compile(r"^what_(MAO_ST|MAO|MA|MO)\s*:")


def main(path):
    cur, block = None, {}
    blocks = []
    for line in open(path, errors="replace"):
        line = line.strip()
        h = HEAD.match(line)
        if h:
            cur = h.group(1)
            continue
        m = ROW.match(line)
        if m and cur:
            block.setdefault(cur, {})[m.group(1)] = float(m.group(3))
            if cur == "MAO_ST" and m.group(1) == "average":
                blocks.append(block)
                block, cur = {}, None
    print(f"{len(blocks)} validation(s) in {path}\n")
    hdr = (f"{'#':>3}  {'axis':>6} {'origin':>6} {'BOTH':>6}   "
           f"{'rot_MA':>6} {'rot_MO':>6} {'rot_BOTH':>8}  "
           f"{'tr_MA':>6} {'tr_BOTH':>7}")
    print(hdr)
    print("-" * len(hdr))
    for i, b in enumerate(blocks):
        g = lambda k, r: b.get(k, {}).get(r, float("nan"))
        print(f"{i:>3}  {g('MA','average'):6.4f} "
              f"{g('MO','average'):6.4f} {g('MAO_ST','average'):6.4f}   "
              f"{g('MA','rotation'):6.4f} {g('MO','rotation'):6.4f} "
              f"{g('MAO_ST','rotation'):8.4f}  "
              f"{g('MA','translation'):6.4f} {g('MAO_ST','translation'):7.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
