# SIVP working-tree update: 17 November 2026

Associated manuscript: **Overall Accuracy Gains Mask Repetition Errors in Supervised Video Counting**, with four main tables and supplementary Tables S1–S15. This note records the public artifact update; it does not record journal acceptance.

## Added material

- `09_qwen25_update/`: 13 run/check records and the English provenance note in Markdown and PDF. These correspond to Online Resource 2, `ESM_2_Qwen25_Provenance.zip`.
- `07_scripts/audit_qwen25.py`: standard-library checks of the added archived predictions and their reported metrics.
- `05_figures/sivp_20261117/Fig1.pdf`–`Fig4.pdf`: the four current main-article figures.
- Updated README, result-to-file map, reproduction scope and v1.0 citation metadata. The original manifest is preserved as `08_history/MANIFEST_v1.0.md`.

The benchmark description is corrected to 60 action labels in total: 59 occur in training and one only in development. The evaluable benchmark remains 3,710 questions over 1,563 videos. This wording correction does not change its unit files or split assignments.

## Qwen2.5 result replacement

The earlier result of 73.0% overall and 9.4% repeated-action accuracy could not be reproduced from the recorded recipe. The current manuscript replaces it with two new runs:

| Run | Overall (%) | Single (%) | Repeated (%) | Answer-one share (%) |
|---|---:|---:|---:|---:|
| Zero-shot | 41.4 | 49.5 | 22.2 | 49.6 |
| repro1: main Table 1 | 78.3 | 97.8 | 32.2 | 84.9 |
| repro2: Section 4.1 | 78.0 | 94.1 | 39.8 | 79.7 |

Both adapted runs improve repeated-action accuracy, with larger gains on single-occurrence questions. They share the recorded hyperparameters and seed argument; the source of their difference was not isolated. The commands, saved settings and predictions support checking these records. Model/adapter weights, hidden states and the training script are not added.

The three historical 60-question-check files concern the earlier `lora_q25_v3` adapter. They are retained to document that earlier check and do not verify the new runs.

## Why the original summaries remain

The v1.0 predictions, aggregate JSONs and root-level figure PNGs remain unchanged so readers can recover the earlier analyses and distinguish them from this update. In particular, `02_predictions/v3_test_q25_lora.jsonl` and `04_summaries/v3_test_q25_lora_summary.json` are historical evidence, not the current Qwen2.5 result. The new results are stored separately instead of overwriting their predecessors.

`MANIFEST.md` now maps the four main tables and Tables S1–S15 to their sources. Existing audit scripts retain their original expected values; the new Qwen2.5 audit is separate. A passing archived-output check does not reproduce model training or validate every interval.

## Citation boundary

The existing DOI [10.5281/zenodo.24187160](https://doi.org/10.5281/zenodo.24187160) describes the **v1.0 archive under the original manuscript title**. `CITATION.cff` continues to describe that archive. It does not describe a new release or include this update's Qwen2.5 records and SIVP figure PDFs. Cite the GitHub repository at a fixed full commit SHA for the later material.

No new release number or DOI is assigned by this update. The full manuscript and cover letter are not added to the public repository.
