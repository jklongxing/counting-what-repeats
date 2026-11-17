# Result-to-file manifest

Updated 17 November 2026 for **Overall Accuracy Gains Mask Repetition Errors in Supervised Video Counting**, with four main tables and supplementary Tables S1–S15. The earlier 14-table map is preserved in `08_history/MANIFEST_v1.0.md`. This working-tree update does not alter the existing v1.0 Zenodo archive.

The root SHA-256 list records file membership and integrity, excluding itself. Summary names below are relative to `04_summaries/`; prediction names are relative to `02_predictions/` unless a full repository path is given. These are evidence mappings, not claims of independent recomputation of every cell or interval.

## Main article

| Item | Evidence |
|---|---|
| Table 1: core configurations | `canonical_results.json`, `head_ablation.json`, `v3_test_lora_bal_summary.json`; corresponding `v3_test_*` and `probe3_lora_test.jsonl` predictions. The new Qwen2.5 LoRA row uses `09_qwen25_update/repro1_test_predictions.jsonl` and its summary, replacing the historical `v3_test_q25_lora` result. |
| Table 2: output/classifier-input controls | `output_calibration.json`, `matched_calibration_and_groups.json`, `calibration_frontier.json`, `extra_controls.json`, `plane_points.json`; `scorehead_test.jsonl` and `probe3_lora_test.jsonl`. |
| Table 3: both-correct paired questions | `same_video_all_splits.json` (test), `extra_controls.json` (`same_question`), benchmark units and corresponding zero-shot/LoRA/readout predictions. Same-question comparisons are descriptive. |
| Table 4: Countix | `countix_refit.json`, `countix_readout.json`, `countix_base_summary.json`, `countix_lora_summary.json`; `countix_base.jsonl`, `countix_lora.jsonl` and `countix_refit_seed0.jsonl` through `seed4.jsonl`. The non-refitted readout is aggregate-only; obtain labels separately. |
| Figure 1: overall/repeated accuracy | `05_figures/sivp_20261117/Fig1.pdf`; configurations in Table 1. |
| Figure 2: adapter-matched readout | `05_figures/sivp_20261117/Fig2.pdf`; `multiseed_probe.json` and text summaries `v3_test_lora_32f`, `v3_test_seed7`, `v3_test_seed8`. |
| Figure 3: frame budgets | `05_figures/sivp_20261117/Fig3.pdf`; `canonical_results.json` and `frame_curves_other_models.json`. |
| Figure 4: single/repeated points | `05_figures/sivp_20261117/Fig4.pdf`; `plane_points.json`. |

## Online Resource 1: Tables S1–S15

| Table | Evidence |
|---|---|
| S1: frame budget sweep | `canonical_results.json` (`frame_curve_qwen3`); `v3_dev_4f.jsonl`, `v3_dev_8f.jsonl`, `v3_dev_16f.jsonl`, `v3_dev_base_32f.jsonl`, `v3_dev_128f.jsonl`, `v3_dev_256f.jsonl`. |
| S2: adapter-matched readout | `multiseed_probe.json`; text predictions `v3_test_lora_32f.jsonl`, `v3_test_seed7.jsonl`, `v3_test_seed8.jsonl`. Readout values summarize five classifier seeds per adapter. |
| S3: accuracy by true count | `01_benchmark/evaluable_units.jsonl`, `v3_test_base_32f.jsonl`, `v3_test_lora_32f.jsonl`, `probe3_lora_test.jsonl`; by-count check in `07_scripts/audit_round2.py`. |
| S4: same-video groups by size | `same_video_groups.json`; these all-correct groups differ from the exhaustive pairs in main Table 3. |
| S5: output calibration | `output_calibration.json`, `matched_calibration_and_groups.json`, `calibration_frontier.json`; free-generation predictions `v3_test_lora_32f.jsonl` and `val_lora_32f.jsonl`. |
| S6: same-question pairs | `extra_controls.json` (`same_question`); paired-answer checks in `07_scripts/audit_round2.py`. |
| S7: reported run settings | The supplementary configuration table and archived scripts including `07_scripts/multiseed_probe.py` and `07_scripts/head_ablation.py`; a configuration record, not a complete training package. Separate Qwen2.5 commands/settings are in `09_qwen25_update/`. |
| S8: supervision across splits | `canonical_results.json`; `v3_test_base_32f.jsonl`, `v3_test_lora_32f.jsonl`, `v3_test_fullft.jsonl`, `v3_dev_base_32f.jsonl`, `v3_dev_lora_32f.jsonl`, `val_base_32f.jsonl`, `val_lora_32f.jsonl`. |
| S9: zero-shot models and priors | `canonical_results.json`; `v3_test_base_32f.jsonl`, `v3_test_qwen25_base.jsonl`, `v3_test_internvl.jsonl`, `v3_test_internvl25.jsonl`; priors use derived training units. |
| S10: same-video pairs across splits | `same_video_all_splits.json` and split-specific predictions. |
| S11: score-input versus hidden-state classifier | `extra_controls.json` (`score_head`); `scorehead_test.jsonl`, `scorehead_val.jsonl`, `probe3_lora_test.jsonl`, `probe3_lora_val.jsonl`. |
| S12: Countix training sources/evaluation | `countix_refit.json`, `countix_readout.json`, `countix_overlap.json` and the final Countix predictions listed for main Table 4. |
| S13: training-subset diagnostic | `probe_ablation.json` (`data_ablation`); a separate fixed-seed diagnostic, not the primary readout result. |
| S14: text/logistic/MLP readout | `head_ablation.json`, `canonical_results.json`; `v3_test_lora_32f.jsonl`, `probe3_lora_test.jsonl`, `val_lora_32f.jsonl`, `probe3_lora_val.jsonl`. |
| S15: nine operating points | `canonical_results.json`, `head_ablation.json`, `plane_points.json`. Main Figure 4 additionally includes calibration and score-space controls. |

Other supplementary diagnostics use `label_risk.json`, `review_analyses.json`, `rescore_mc_and_ordering.json`, `mc_readout.json`, `synth_test_eval_summary.json` and corresponding predictions/scripts. Their input requirements differ; see `REPRODUCIBILITY.md`.

## Online Resource 2: Qwen2.5 update

`09_qwen25_update/` contains the 13 run/check records and English PDF note supplied in `ESM_2_Qwen25_Provenance.zip`, plus an editable Markdown copy of the note. The new test records cover 577 questions (406 single, 171 repeated). `07_scripts/audit_qwen25.py` recomputes their metrics.

| Run | Overall (%) | Single (%) | Repeated (%) | Answer-one share (%) |
|---|---:|---:|---:|---:|
| Baseline | 41.4 | 49.5 | 22.2 | 49.6 |
| repro1 (main Table 1) | 78.3 | 97.8 | 32.2 | 84.9 |
| repro2 (Section 4.1) | 78.0 | 94.1 | 39.8 | 79.7 |

Both adapted runs share the recorded hyperparameters and seed argument; the source of their difference was not isolated. The three `repro_check_60.txt`, `repro60_q25v3_records.jsonl` and `repro60_records.log` files concern the earlier `lora_q25_v3` adapter and do not verify repro1 or repro2.

## Historical evidence

The earlier Qwen2.5 LoRA result of 9.4% repeated-action accuracy could not be reproduced from the recorded recipe and was replaced. Its files, including `v3_test_q25_lora.jsonl` and `v3_test_q25_lora_summary.json`, remain historical records, not the current Qwen2.5 result. Likewise, `dev2_*`, `test_*`, `probeL_*` and `probeQ25L*` prefixes do not identify the current main configurations by themselves.

Original aggregate JSONs remain unchanged so readers can recover the v1.0 analyses. They have not been rewritten to adopt the current table layout or new Qwen2.5 results. The original two audits also retain their numerical logic and expected values. Use this map and `06_documents/SIVP_20261117_UPDATE.md` to identify current evidence.

For Countix refitting, use `countix_refit_seed0`–`seed4` and final `countix_refit.json`: 56.47% mean, reported as 56.5%. The 57.8% in an earlier report is superseded. Countix JSONL files omit upstream labels/text; follow `06_documents/COUNTIX_REPRODUCTION.md`.

The withdrawn same-question bootstrap stays in `08_history/` and supplies no active reported interval. Original PNGs directly under `05_figures/` retain the earlier v1.0 figure numbering. Current main figures are the four PDFs in `05_figures/sivp_20261117/`. Manuscript drafts and cover letters are not distributed here.
