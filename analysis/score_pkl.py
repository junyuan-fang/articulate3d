#!/usr/bin/env python
"""Score any preds.pkl with the official val criteria. Usage: score_pkl.py <pkl>"""
import sys
import build_matches as bm
import benchmark.evaluate_semantic_instance as esi

bm.PRED_PKL = sys.argv[1]
matches, _, _ = bm.build()
print(f"scenes: {len(matches)}   ({sys.argv[1]})")
for name, crit in [("AP50", esi.match_criteria_M_pred),
                   ("AP50_axis", esi.match_criteria_MA_pred),
                   ("AP50_origin", esi.match_criteria_MO_pred),
                   ("AP50_axis_origin", esi.match_criteria_MAO_pred_standard)]:
    ap = esi.compute_averages(esi.evaluate_matches(matches, crit))
    print(f"  {name:18s} {ap['all_ap_50%']:.4f}")
