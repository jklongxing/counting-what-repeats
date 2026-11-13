> Historical document. See ../README.md and the current top-level MANIFEST.md / AUDIT_REPORT.md. This is not a current certification.

# Independent audit of the reported numbers

Date: 2026-11-11. Scope: `JVCI_Counting_What_Repeats.docx` (19 pages, 10 tables, 8 figures).

## Method

Two scripts recompute the reported values **from the per-question predictions** in
`02_predictions/`, without reading `04_summaries/canonical_results.json`. They rebuild each
metric from scratch, then compare the result with the number printed in the manuscript and
flag any difference above 0.06 percentage points.

- `audit_recompute.py` covers Tables 4.1, 4.2, 4.3, 4.4, 4.5 and the size of the evaluated
  universe.
- `audit_round2.py` covers Table 4.7 (accuracy by true count), Table 4.9 (single versus
  repeated plane), Table 4.10 (same-video pairs), the human-written questions, and every
  arithmetic identity asserted in the abstract and text.

Both scripts are in `07_scripts/`. To reproduce:

```
python audit_recompute.py
python audit_round2.py
```

## Result

Final state: **0 flags in both audits.** Every table cell that the scripts check reproduces
from the predictions, and every stated difference equals the difference of the unrounded
ratios.

## What the audit found and what was changed

| Finding | Severity | Action |
|---|---|---|
| `benchmark3_units.jsonl` contains **3,745** units (the constructed set), while the paper evaluates **3,710** (those with a complete 32-frame grid). The bundle lacked the evaluated list, so a naive recomputation silently used 3,745 and disagreed with Table 4.2 by up to 0.4 points. | packaging, not a result error | added `01_benchmark/evaluable_units.jsonl` (3,710 units: 2,555 / 578 / 577) and made both audits use it |
| Table 4.1 and Table 4.9: InternVL3.5-8B single-occurrence accuracy was printed as **66.8%**. The exact value is 271/406 = 66.749%, which rounds to **66.7%**. | **real error, 0.1 point** | corrected to 66.7 in both tables |
| Abstract, contribution 3 and Section 4.3: the repeated-question gain over the action prior was printed as **26.4 points**. Exact: 43.8596 − 17.5439 = 26.316 → **26.3**. | **real error, 0.1 point** | corrected to 26.3 in all three places |
| Two cells sit exactly on a rounding boundary: VAL readout single-occurrence accuracy is 414/480 = 86.25% and is printed as 86.3. | convention, not an error | added a sentence to Section 4.0 stating that percentages are rounded half-up and that a stated difference is computed from unrounded ratios, so it can differ by 0.1 from the difference of two rounded entries |
| The audit initially reported a mismatch on "readout − LoRA = 12.9" because it subtracted the rounded table entries (56.7 − 43.9 = 12.8). The unrounded difference is 56.725 − 43.860 = 12.87. | audit artefact | the audit now uses unrounded operands, and the manuscript states the convention |

No other discrepancy was found. In particular, the four zero-shot models, the six frame
budgets, all supervision and readout rows, the three adapter seeds, the human-written rows,
the by-count table, the plane table and the same-video rates all reproduce.

## What this audit does not verify

Being explicit about the boundary of the check:

1. **The model outputs themselves.** The audit trusts the saved predictions. It cannot prove
   that the model produced them; that would require re-running inference. The predictions do
   carry the raw answer string next to the parsed integer, so a reader can see the parse.
2. **The pairwise tests and intervals.** These are produced by the bootstrap in
   `canonical_eval_v2.py` and `matched_calibration.py` and were not independently
   re-implemented. Their inputs are the same prediction files that the audit does verify.
3. **The label construction.** The audit checks the *use* of the labels, not that the builder
   assigned them correctly. The builder's own filter was re-run separately
   (`diagnose_labels_v2.py`, output `label_risk.json`: 10 of 3,710 units affected), which is a
   consistency check against the raw annotations, not an independent re-annotation.
4. **Training.** The adapters were trained once per seed; the audit does not re-train.
5. **Figures.** They are rendered from the audited numbers by `plot_pil.py`.

## Honest summary

Two numbers were wrong by 0.1 point and are now fixed. One packaging gap (the evaluated-unit
list) could have made an external reader's recomputation disagree with the paper, and is now
closed. The remaining values reproduce exactly under the stated metric, and the metric itself
is defined in one place and applied by one script.

## Fourth round: three defects found by external review and fixed

| Defect | What was wrong | Fix |
|---|---|---|
| Countix table mixed two metrics | The two text rows were raw exact match (17.7, 41.0) while the refitted head was four-way (56.7), so the stated gain of 15.7 points was not comparable to the in-domain 12.9 | all four rows now four-way: 30.7, 41.2, 34.0, 57.8; the gain over the text model is 16.6 points |
| One video was in both the Countix fit and the evaluation set | s5stimPX3h0_cut0 appeared in the 1,292 fitted units and in the 476 evaluated units | excluded from fitting; accuracy after exclusion is 57.8 +/- 0.7 (five seeds), the overlap file is archived, and the actual train/evaluation video overlap is now empty |
| A paired significance test that did not hold | the same-question comparison (40.8 vs 30.0) carries dependent pairs, and the bootstrap we used either resampled dependent pairs or split pairs across two clusters | p value and intervals withdrawn; the comparison is reported descriptively, and the table no longer has a CI column |

Also corrected: the Countix section no longer claims that the hidden state is intrinsically better than the answer scores on a new corpus, since the refitted head received target labels and a restricted output set that the text model did not; a constant answer of 2 scores 48.3% on that subset and is stated as the scale. The section states that the question template is ours, not Countix's. Section 4.11 now gives 4.8 points rather than 4.7 for the VAL single-occurrence cost, and the contribution list no longer says "equal capacity".

## What this says about the process

These were avoidable. The metric mix-up came from reading two different result files without normalising the metric; the leakage came from never exporting the actual fitted ID list; the statistics came from picking a resampling scheme by convenience rather than by the dependence structure. The audit scripts do not catch any of the three, because they check that numbers reproduce from predictions, not that the comparison is well posed. A comparison audit (same metric, same units, disjoint sets, stated dependence structure) is the missing layer, and it is now part of the checklist for any new table.
