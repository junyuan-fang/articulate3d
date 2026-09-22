# Why a "perfect GT mask" oracle does not score 1.0 on Articulate3D val

Metric: `AP50_axis_origin` = `match_criteria_MAO_pred_standard` = **MAO_ST**, the leaderboard metric.
Evaluator: `/data/challenge/USDNet/benchmark/evaluate_semantic_instance.py`, **unmodified** for every
number below except the explicitly labelled intervention (Step 3), which uses a generated copy at
`/data/challenge/analysis/_evaluate_semantic_instance_clipped.py`.

Script: `/data/challenge/analysis/oracle_gap.py` · machine-readable results:
`/data/challenge/analysis/oracle_gap_results.json` · raw log: `/data/challenge/analysis/logs_oracle_gap.log`

Reproduce:
```bash
cd /data/challenge/analysis
PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis \
$HOME/.conda/envs/usdnet/bin/python -u oracle_gap.py
```

**Coverage.** 42/42 validation scenes, 390/390 valid GT instances (236 rotation, 154 translation).
One scene (`a980334473`) contains 0 valid GT instances; the evaluator returns `nan` for it and
`np.nanmean` drops it — no loss. Oracles (c)/(d) and the model baseline cover the 41 scenes present
in `preds.pkl` (that same empty scene is missing from the pickle).

---

## Step 1 — which oracle produces what

| # | oracle | AP50 | AP50_axis | AP50_origin | **MAO_ST** |
|---|---|---|---|---|---|
| (a) | GT mask + GT class + GT axis + GT origin, all conf = 1.0 | 1.0000 | 0.6516 | 1.0000 | **0.6516** |
| (b) | same as (a), conf ~ U(0,1) | 1.0000 | 0.5717 | 1.0000 | **0.5717** |
| (b2) | same as (a), conf descending 1.0→0.01 | 1.0000 | 0.5893 | 1.0000 | **0.5893** |
| (c) | GT mask + GT class + **predicted** axis/origin | 1.0000 | 0.6612 | 0.7433 | **0.4227** |
| (d) | **predicted** mask + GT axis/origin (275 queries replaced) | 0.5716 | 0.3860 | 0.5473 | **0.3511** |
| — | model baseline (`eval_mov_val_einsum/.../preds.pkl`) | 0.5716 | 0.4432 | 0.4495 | 0.3215 |

**None of the four reproduces 0.9448.** The prioritised oracle (a) lands at **0.6516**.
See "What we could not reproduce" below for the search that was done.

Note (d): pasting the *exact GT axis* onto the model's own predictions **lowers** `AP50_axis`
from 0.4432 to 0.3860. Being exactly right is punished. That is the whole story, in one line.

---

## Step 2 — decomposition of the 1.0 → 0.6516 gap

### The mechanism (H1), confirmed

`match_criteria_MA_pred`, lines 133-150:

```python
deriavation = np.dot(axis_ref, axis_checked) / (np.linalg.norm(axis_ref) * np.linalg.norm(axis_checked))
deriavation = np.abs(deriavation)
is_axis = np.arccos(deriavation) < np.deg2rad(axis_th)
```

For `axis_ref == axis_checked` (a bit-exact GT copy) this expression evaluates to
`1.0000000000000002` (= 1 + 1 ULP; measured `max_dev = 1.0000000000000002`) whenever the rounding of
`dot` and of `norm*norm` disagree. `np.arccos(1+eps)` is `nan`, and `nan < th` is `False`, so a
**perfect axis is scored as a miss**.

The failure is charged twice, because the evaluator calls the criterion in both directions:

* GT loop `match_criteria(pred, gt, th, False)` fails → `found_match` stays False → **hard false negative** (kills recall);
* pred loop `match_criteria(gt, pred, th, True)` fails → `found_gt` stays False, `proportion_ignore = 0 ≤ th` → **false positive** (kills precision).

`np.dot(a,b)` and `np.dot(b,a)` are not bit-identical for every vector, so the two counts differ by 1
per class — an instance can be an FP without being an FN.

### Measured counts (instances, out of 390)

| | rotation (N=236) | translation (N=154) | total |
|---|---|---|---|
| NaN in GT loop → hard FN | **10** | **78** | 88 (22.6%) |
| NaN in pred loop → FP | **11** | **79** | 90 (23.1%) |
| TP | 226 | 76 | 302 |

Arithmetic check against the evaluator's own AP (single unique confidence, so
`AP = ½·r·(1+p)` exactly — see H5):

* rotation: r = 226/236 = 0.957627, p = 226/237 = 0.953586 → AP = **0.935401** (evaluator: 0.9354)
* translation: r = 76/154 = 0.493506, p = 76/155 = 0.490323 → AP = **0.367736** (evaluator: 0.3677)
* mean = **0.651569** (evaluator: 0.6516) ✓

### Contribution to the gap

| mechanism | AP points lost | share of the 0.3484 gap |
|---|---|---|
| H1 arccos NaN, **translation** class | 0.3161 | **90.7 %** |
| H1 arccos NaN, **rotation** class | 0.0323 | 9.3 % |
| everything else (H2, H3, H4, H5, H6-intrinsic, H7) | 0.0000 | 0 % |

### Why translation is hit 5× harder (this is H6, and it is a *consequence* of H1, not a separate cause)

| | exactly axis-aligned GT axes (one component ±1.0, two exactly 0) | NaN rate |
|---|---|---|
| rotation | **206 / 236 (87.3 %)** | 10/236 = 4.2 % |
| translation | **0 / 154 (0 %)** | 78/154 = 50.6 % |

For an axis-aligned vector `dot(a,a)` and `norm(a)*norm(a)` are both exactly 1.0, so the quotient is
exactly 1.0 and `arccos` is fine. Rotation hinges in Articulate3D are overwhelmingly gravity-aligned
and stored as literal `(0,0,±1)`; translation axes are generic horizontal azimuths with three
non-trivial mantissas, so ~half of them round above 1. The class-averaged AP therefore hides a
5× asymmetry, and MAO_ST is dragged down almost entirely by the `translation` column.

### Hypotheses individually ruled out

| H | claim | measurement | verdict |
|---|---|---|---|
| **H2** | GT instances that can never be matched | `instance_id < 1000`: **0**; `vert_count < min_region_size(=1)`: **0** (smallest GT instance = 16 pts); GT ids missing from the articulation h5: **0**; h5 entries missing from the GT txt: **0**. The 42 "invalid class" ids are the single id `1` (label 0 = unannotated) present once per scene — never a GT instance. | ruled out (0 AP) |
| **H3** | duplicate/overlapping GT instances | the GT txt stores **one integer per point**, so GT instances are disjoint by construction; empirically `AP50 = 1.0000` for oracle (a), i.e. every GT-copied mask matched its own GT at IoU 1.0 and nothing else. | ruled out (0 AP) |
| **H4** | degenerate GT articulation | zero-length axes: **0**; non-finite axes: **0**; non-finite origins: **0**; all 390 axis norms lie in [1−2ulp, 1+2ulp] (362/390 exactly 1.0). The processed h5 carries only `axis/origin/sem_id/inter_mask` — no `needsReview` field reaches the evaluator; in the raw annotations for the 42 val scenes, 258 `needsReview` fields exist and **0 are true**. | ruled out (0 AP) |
| **H5** | AP-curve mechanics with tied confidences | verified by hand-reimplementation of the exact integration. With one unique score: `recall_for_conv = [r, r, 0, 0]`, `stepWidths = [r/2, r/2]`, `AP = ½·r·(1+p)`. All-TP, no FN, no FP → **exactly 1.0** (also 1.0 with random scores). Sanity points: 99TP+1FN → 0.99; 99TP+1FP → 0.995; both → 0.98505. Confidence values are *not* the cause: oracles (a)/(b)/(b2) all reach 1.0000 once H1 is fixed. | ruled out (0 AP) |
| **H6** | per-class averaging drags the mean | Both classes are present and non-NaN. Translation *is* dragging the mean (0.3677 vs 0.9354), but that is entirely the H1 NaN rate, not an averaging artefact — after the clip patch both classes are exactly 1.0000. | not an independent cause |
| **H7** | mask space mismatch (faces vs vertices, voxelization) | for all 42 scenes `len(gt_ids) == len(processed .npy points) == preds.pkl mask rows`; **0 mismatches**. `AP50 = 1.0000` for oracle (a) confirms it end to end. | ruled out (0 AP) |

---

## Step 3 — intervention (on a **copy** of the evaluator)

`oracle_gap.py` writes `/data/challenge/analysis/_evaluate_semantic_instance_clipped.py`, a verbatim
copy of the official evaluator with exactly one line changed:

```python
-    deriavation = np.abs(deriavation)
+    deriavation = np.clip(np.abs(deriavation), -1.0, 1.0)
```

The original evaluator is untouched.

| evaluator | AP50 | AP50_axis | AP50_origin | MAO_ST |
|---|---|---|---|---|
| official | 1.0000 | 0.6516 | 1.0000 | 0.6516 |
| **+ `np.clip`** | 1.0000 | **1.0000** | 1.0000 | **1.0000** |

The one-line patch recovers **100 % of the gap** (+0.3484). Nothing else is needed. This is the
definitive confirmation that H1 is the sole mechanism for oracle (a).

---

## Sensitivity: the oracle's score is decided by the last bits of the axis

Same oracle (a), only the float representation of the *copied* GT axis changed. Masks, classes,
origins and confidences are identical throughout, and `AP50 = 1.0000` in every row.

| axis representation | NaN→FN (rot / transl) | MAO_ST | rotation | translation |
|---|---|---|---|---|
| `%.3f` … `%.5f` | 0 / 0 | **1.0000** | 1.0000 | 1.0000 |
| float16 round-trip | 0 / 0 | **1.0000** | 1.0000 | 1.0000 |
| `%.6f` (= what `transfer_preds_files.py` writes, `fmt='%f'`) | 0 / 1 | 0.9951 | 1.0000 | 0.9903 |
| `%.7f` | 1 / 5 | 0.9743 | 0.9937 | 0.9549 |
| `%.8f` | 12 / 70 | 0.6718 | 0.9212 | 0.4225 |
| `%.10f` | 18 / 92 | 0.5854 | 0.8885 | 0.2823 |
| `%.11f` | 21 / 93 | **0.5741** (worst) | 0.8723 | 0.2760 |
| float32 → float64 | 13 / 51 | 0.7368 | 0.9170 | 0.5566 |
| **verbatim float64 copy** (oracle a) | 10 / 78 | **0.6516** | 0.9354 | 0.3677 |
| verbatim, `a/‖a‖` renormalised | 9 / 79 | 0.6505 | 0.9396 | 0.3613 |
| verbatim, stored as float32 array | 20 / 120 | 0.5056 | 0.8765 | 0.1348 |

Practical consequence: rounding a submission's axes to 5 decimals is a strictly-free +0.35 AP for a
perfect predictor, and the difference between a 6-decimal and an 8-decimal submission file is
0.9951 vs 0.6718. This is a property of the metric implementation, not of any model.

---

## What we could **not** reproduce: 0.9448

We could not construct any oracle scoring 0.9448. Tested and measured (all in
`oracle_gap_results.json`):

* the four requested oracles (a)-(d): 0.6516 / 0.5717 / 0.4227 / 0.3511;
* 20 axis float representations of oracle (a): the reachable set is
  {1.0000, 0.9951, 0.9743, 0.7368, 0.6718, 0.6516, 0.6505, 0.6432, 0.6282, 0.6140, 0.6015, 0.5854, 0.5741, 0.5056, …}
  — no value near 0.9448 (nearest above: 0.9743, nearest below: 0.9082);
* six "query-slot" oracles (GT written into the best-IoU query slot of `preds.pkl`, which loses 30
  GT instances to slot collisions and caps `AP50` at 0.9082/0.9100):
  0.6226 / 0.6568 / 0.9034 / 0.9052 / **0.9082** / 0.9082.

Arithmetic constraint. For a GT-copy oracle with tied confidences the evaluator gives
`AP = ½·r·(1+p)` per class. If every failing instance is both an FN and an FP (which is what the
NaN bug produces), an exhaustive search over all `(k_rot, k_transl) ∈ [0,236]×[0,154]` yields
**no** pair whose class mean rounds to 0.9448. The value *is* reachable if failures are hard FNs
**without** an accompanying FP, i.e. if some GT instances simply have no prediction emitted:
`(k_rot, k_transl) = (0, 17)` gives 0.944805 and `(23, 2)` gives 0.944778. We found no natural
construction that drops exactly 17 translation instances, so this remains an **unconfirmed
arithmetic possibility**, not a finding.

Most likely explanations for the user's 0.9448, none verifiable from this repo:

1. the oracle was scored through a route that alters the axis floats (a submission-file round trip,
   a float cast, an explicit normalisation), placing it somewhere on the sensitivity curve above —
   the curve passes through 0.9743 and 0.9082 but we did not hit 0.9448 with any tested route;
2. the oracle emitted fewer predictions than GT instances (recall-only loss);
3. the number came from the challenge server (dev phase) rather than the local harness, whose GT we
   cannot inspect.

**Honest bottom line:** 0.9448 is not reproducible from the repo as it stands. What *is* established
beyond doubt is the mechanism — the `arccos` domain overflow — and its magnitude: it costs a
bit-exact oracle **0.3484 MAO_ST** (1.0000 → 0.6516), 90.7 % of that on the `translation` class, and
a single `np.clip` recovers all of it.

---

## Statement for the paper (measured, safe to quote)

> The evaluator computes the axis agreement as `arccos(|⟨a,a_gt⟩| / (‖a‖‖a_gt‖))`. The argument is
> not clamped, so for a bit-exact prediction the quotient evaluates to `1+2^-52` and `arccos`
> returns `NaN`; since `NaN < θ` is `False`, an exactly correct axis is scored as a miss, and the
> evaluator's two-directional matching charges it as both a false negative and a false positive.
> A GT oracle (GT masks, classes, axes and origins, unit confidences) therefore scores
> MAO_ST = 0.6516 instead of 1.0 on validation: 88 of 390 GT instances (22.6 %) are rejected in the
> GT-matching direction and 90 in the prediction-matching direction. The effect is strongly
> class-dependent — 206/236 rotation axes are stored as exact canonical vectors `(0,0,±1)`, for which
> the quotient is exactly 1.0, whereas 0/154 translation axes are, so the NaN rate is 4.2 % for
> rotation versus 50.6 % for translation (per-class AP 0.9354 vs 0.3677). Clamping the argument
> (`np.clip(·, -1, 1)`) restores the oracle to exactly 1.0000. As a corollary, submitted axes should
> be rounded (≤5 decimals) rather than submitted at full precision: on our oracle this is worth up
> to +0.35 MAO_ST, and pasting exact GT axes onto the model's own predictions *lowers* AP50_axis
> from 0.4432 to 0.3860.
