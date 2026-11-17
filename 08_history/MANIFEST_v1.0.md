# Release manifest

Prepared 13 November 2026 for `v1.0`. Associated manuscript: **Supervision and Readout for Action-Conditioned Counting in Natural Videos**, submission v11.

## File inventory

| Directory | Files |
|---|---:|
| `01_benchmark/` | 8 |
| `02_predictions/` | 119 |
| `03_scores/` | 4 |
| `04_summaries/` | 108 |
| `05_figures/` | 9 |
| `06_documents/` | 2 |
| `07_scripts/` | 30 |
| `08_history/` | 5 |

Root files contain the README, licenses, citation, manifest, validation report and SHA-256 list. The complete checksum list is authoritative for membership; it excludes itself.

## Current manuscript map

All unqualified summary names below are in `04_summaries/`; prediction names are in `02_predictions/`. A mapping records provenance, not a claim that every cell has an independent runnable check.

| Manuscript item | Archived evidence | Reproduction route |
|---|---|---|
| Table 1, configurations | Manuscript configurations and archived experiment scripts | Descriptive configuration record; not a numerical audit |
| Table 2, zero-shot | `canonical_results.json`, `v3_test_*` predictions, derived labels | `audit_recompute.py` for its explicit rows |
| Table 3, frame budget | `v3_dev_{4,8,16,128,256}f`, `v3_dev_base_32f`, `canonical_results.json` | `audit_recompute.py` |
| Table 4, supervision | `v3_test_lora_32f`, `v3_test_fullft`, base/VAL predictions, canonical summary | `audit_recompute.py`; prior definitions in canonical script |
| Table 5, readout | `probe3_lora_*`, `head_ablation.json`, canonical summary | Audit checks for its explicit configurations; ablation summary for the others |
| Table 6, adapter seeds | `multiseed_probe.json`, seed-specific prediction files | Selected adapter checks in first audit; archived multiseed analysis |
| Section 4.5, human-written questions | `mc_units.jsonl`, `mc_*`, `rescore_mc_and_ordering.json`, `mc_readout.json` | Both audits' implemented checks |
| Section 4.6, synthetic data | `synth_test_eval_summary.json`, `v3_test_synth_32f*` | Archived analysis and predictions |
| Table 7, true count | Evaluable units and zero-shot/LoRA/readout predictions | `audit_round2.py` |
| Section 4.8, supporting slots | `review_analyses.json` | Original script requires source evidence metadata |
| Table 8, single/repeated plane | `canonical_results.json`, `head_ablation.json`, `plane_points.json` | `audit_round2.py` for its explicit points |
| Tables 9–10, same-video pairs/groups | `same_video_all_splits.json`, `same_video_groups.json`, predictions | Selected pair checks in second audit; archived group analyses |
| Table 11, calibration | `output_calibration.json`, `matched_calibration_and_groups.json`, `calibration_frontier.json` | Archived calibration analyses; implemented checks in second audit |
| Tables 12–13, same-question and score head | `extra_controls.json`, `scorehead_test.jsonl`, `scorehead_val.jsonl` | `audit_round2.py`; no active same-question CI |
| Table 14, Countix | `countix_refit.json`, `countix_readout.json`, released project predictions | External labels + `recompute_countix.py` for text/refit rows; non-refitted readout is aggregate-only |
| Section 3 label consistency check | `label_risk.json` | Original consistency-check script requires source annotations |
| Figures 1–8 / graphical abstract | `05_figures/` | Exact v11 submission PNGs; layout source for Figures 1/5 added separately |

## Run and history conventions

Main configurations include `v3_test_base_32f`, `v3_test_lora_32f`, `v3_test_fullft`, `probe3_lora_test`, adapter seeds 7/8, the `v3_dev_*` frame sweeps, and `val_*` / `probe3_lora_val`. Earlier `dev2_*`, `test_*`, `probeL_*`, `probeQ25L*` files are retained provenance, not replacements for the mapped final runs.

For Countix refitting use the five `countix_refit_seed0`–`seed4` files and the final `countix_refit.json`. The historical audit report's 57.8% is superseded; current mean is 56.47% (56.5% in the manuscript). Published JSONL records retain project predictions but omit upstream Countix labels/text; see the reproduction note.

The withdrawn same-question bootstrap has moved to `08_history/`. It is not the method used for a reported interval. No old manuscript or internal review memo is part of this release.

## Immutable evidence and preparation changes

Perception Test unit, prediction and score files, and all copied aggregate-result JSONs, are byte-identical to the local source archive. The only record projections are the declared Countix JSONL field removals. Figure files match the v11 submission package. Original source files and the manuscript have not been edited.

The two ready-to-run audit scripts change only their bundle-root path. Other research scripts retain their historical paths and dependencies. Read `REPRODUCIBILITY.md` before using them.
