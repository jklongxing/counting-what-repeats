# Release validation

Prepared 13 November 2026. Scope: this repository copy and its archived-output checks. This is not a new training/inference experiment or a claim of independent human annotation.

## Executed checks

| Check | Observed result |
|---|---|
| `audit_recompute.py` | 75 printed `ok` checks; final message says all checked values reproduce; no mismatch flags |
| `audit_round2.py` | 46 printed `ok` checks; final `PROBLEMS: 0` |
| Countix helper against existing local archived labels | Text rows 30.672269% and 41.176471%; five refit seeds mean 56.470588%, population SD 0.952587 percentage points |
| Perception Test evidence / score / copied summary files | Byte-identical to original bundle |
| Countix release projection | Row count/order, source IDs and numerical predictions preserved; upstream label/text fields excluded |
| Figures | All nine PNGs byte-identical to v11 submission package |
| Original bundle | All 309 original file hashes unchanged |
| Published Python source | Parsed successfully; no syntax errors |
| Common private-key / access-token patterns | No matches in checked release text files |

The audit scripts' numerical logic and expected values were not changed. Their root path alone was repaired. Zero flags applies to the checks those scripts implement, not to every claim, confidence interval or manuscript sentence.

## Coverage limits

No models were loaded; no training, inference, label generation or upstream media download was performed. The audit scripts trust archived model outputs and labels. Their agreement does not establish label correctness or identify the mechanism behind readout gains. Bootstrap intervals and training are not independently reimplemented here. The Countix helper was tested with existing local source labels, which are not included in the public copy; a fresh upstream identifier mapping was not verified. The non-refitted Countix readout row remains aggregate-only in this bundle.

`08_history/` holds earlier reports and the withdrawn same-question bootstrap. Those historical reports contain superseded values and must not be used as the validation status of this release. Run `python 07_scripts/verify_release.py` to verify the final SHA-256 manifest.
