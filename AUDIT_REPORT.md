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

## 17 November 2026 working-tree update

The original report above describes v1.0. The following checks were run on the later update:

| Check | Observed result |
|---|---|
| Original `audit_recompute.py` | All checked values reproduce within 0.05 percentage points; exit 0 |
| Original `audit_round2.py` | `PROBLEMS: 0`; exit 0 |
| New `audit_qwen25.py` | Three 577-question runs, identical unit IDs and ground truth; all implemented summary checks pass |
| New Qwen2.5 records and English notes | All 15 files match the submission's provenance directory byte for byte |
| New four main figures | Match the submission's vector PDFs byte for byte |
| Added provenance text | No private-key, credential, or embedded-password payload found in the checked text |

The full arithmetic report is [QWEN25_AUDIT_20261117.json](06_documents/QWEN25_AUDIT_20261117.json). It records baseline/repro1/repro2 overall accuracies of 41.4/78.3/78.0%, single-occurrence accuracies of 49.5/97.8/94.1%, and repeated-action accuracies of 22.2/32.2/39.8%. Baseline predictions include 173 zeros; these are errors in the four-way 1/2/3/4+ metric and are not clamped to one. Historical 60-question records are excluded from the new-run evidence.

The original benchmark, prediction, score and aggregate-summary files are retained unchanged. These checks verify arithmetic and file consistency, not training or inference reproduction. The existing v1.0 DOI remains unchanged and does not cover these additions. The updated SHA-256 manifest can be verified with `07_scripts/verify_release.py`.

The repository attributes preserve released bytes across platforms; automatic line-ending conversion is disabled. A commit archive was verified against the SHA-256 manifest.
