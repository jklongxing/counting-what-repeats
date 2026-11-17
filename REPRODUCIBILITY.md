# Reproducing the archived results

## Ready-to-run checks

Run the four commands in `README.md` from any working directory by supplying the corresponding script path. The scripts locate the repository relative to their own files. The original `audit_recompute.py` and `audit_round2.py` retain their v1.0 numerical logic and expected values; only their bundle-root path was adapted for the original release. The added `audit_qwen25.py` checks the November update separately.

`audit_recompute.py` checks the four-model comparisons, selected frame/supervision/readout configurations, adapter runs, human-written-question checks and evaluated-unit counts implemented in that script. `audit_round2.py` checks by-count accuracy, selected single/repeated points, paired-answer checks and arithmetic identities implemented in it. Neither script independently validates model generation, upstream labels, training or every bootstrap interval. `AUDIT_REPORT.md` records the observed outputs rather than treating a successful process exit as proof of zero mismatches.

`audit_qwen25.py` checks the new baseline, repro1 and repro2 test predictions in `09_qwen25_update/`, including their question identities, counts, four-way accuracy and answer-one share. It checks archived records rather than rerunning training or inference. The historical 60-question check in that directory concerns the earlier `lora_q25_v3` adapter, not repro1 or repro2. The English provenance note gives the saved commands/settings and explains the replacement of the earlier 9.4% repeated-action result.

## Inputs and eligibility

`benchmark3_units.jsonl` contains 3,745 constructed units. `evaluable_units.jsonl` fixes the 3,710-unit main analysis set. `val_units.jsonl` contains 960 constructed VAL records, while the archived main VAL analysis uses 959 eligible records. The original canonical script applies source-frame availability checks; do not replace that step with an unfiltered read of all constructed records. The released prediction audits preserve their original matching behavior.

The 3,710 evaluable units span 1,563 videos and 60 action labels. Training contains 59 labels; one additional label appears only in development. The 577-question test set has 406 single-occurrence and 171 repeated-action questions. Four-way accuracy uses count buckets 1, 2, 3 and 4+, rather than raw exact match for counts above four.

## Archived research scripts

The remaining scripts in `07_scripts/` are provenance records of the experiment code. They are not an install-and-run training platform. Some use absolute paths from the original server or expect weights, hidden states, source annotations or videos that are not included here. In particular, `canonical_eval_v2.py` requires NumPy/SciPy, source frames and an explicitly configured output directory. Read a script and adapt its data/model/output paths before running it. Do not point its outputs at the frozen release files.

`plot_pil_v11.py` and `04_summaries/plot_data_v11.json` preserve the source of the v11 Figure 1 and Figure 5 layout corrections. Its entry point generates those two figures into `generated/figures_v11/`, leaving release figures unchanged. It needs Pillow; font rendering depends on installed fonts. The archived PNGs in `05_figures/` are the authoritative files supplied with manuscript v11. Figure 6 requires the upstream source frames to rebuild.

The current SIVP main figures are `05_figures/sivp_20261117/Fig1.pdf` through `Fig4.pdf`. The historical plotting commands above do not generate these PDFs. `MANIFEST.md` maps their data to the archived summaries.

## Scope of what is absent

No model or adapter weights, hidden-state caches, full source videos or extracted-frame directory are bundled. The release supports checking archived outputs; reproducing training and inference requires the resources and configurations described in the manuscript. Countix labels are obtained separately as described in `06_documents/COUNTIX_REPRODUCTION.md`.

## Historical material

Earlier run files remain for provenance; prefixes `dev2_*`, `test_*`, `probeL_*` and `probeQ25L*` are not the main reported configurations. The withdrawn same-question bootstrap is isolated in `08_history/`, is not an active analysis command, and must not supply intervals for the descriptive comparison. Historical reports do not certify the current manuscript. Current mapping and release checks are in `MANIFEST.md` and `AUDIT_REPORT.md`.

The earlier `v3_test_q25_lora` prediction/summary files are retained as history after their result could not be reproduced from the recorded recipe. Use `09_qwen25_update/repro1_*` for the current main Table 1 row and `repro2_*` for the second run in Section 4.1. Original aggregate JSONs remain unchanged; they are not a regenerated summary of the new manuscript. See `06_documents/SIVP_20261117_UPDATE.md`.

The existing DOI `10.5281/zenodo.24187160` identifies v1.0 only. The new Qwen2.5 records, audit and SIVP figure PDFs must be identified by the GitHub commit used; they are not included in that earlier DOI archive.
