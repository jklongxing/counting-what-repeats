# Counting what repeats

Evaluation artifacts for **Overall Accuracy Gains Mask Repetition Errors in Supervised Video Counting**, by **Quan Ji**, Tianjin Electronic Information College. This working tree includes the 17 November 2026 update for the SIVP submission.

This repository contains the derived benchmark, archived model predictions, answer scores, aggregate results, figures and analysis scripts. The main benchmark has 3,710 evaluable questions over 1,563 videos and **60 action labels: 59 occur in training and one only in development**. It separates single-occurrence from repeated-action questions. Standard Qwen3-VL LoRA raises overall four-way accuracy from 49.2% to 77.6%, while repeated-action accuracy changes from 42.7% to 43.9%. A hidden-state readout raises the latter to 56.7%, with single-occurrence accuracy decreasing from 91.9% to 88.2%.

See [MANIFEST.md](MANIFEST.md) for the current four main tables and Tables S1–S15, and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the checks supported by these files.

## November 2026 update

- `09_qwen25_update/` adds baseline and two new Qwen2.5-VL-7B runs, their commands, saved settings, predictions and English provenance note. The earlier 9.4% repeated-action result could not be reproduced and is not used in the current manuscript. The new runs reach 32.2% and 39.8%, against a 22.2% zero-shot baseline. Both use the same recorded hyperparameters and seed argument; the source of their difference was not isolated.
- `05_figures/sivp_20261117/` contains the four current main-article vector figures. The PNGs directly under `05_figures/` remain the earlier v1.0 figures.
- The original v1.0 predictions and aggregate summaries remain unchanged for provenance. [The update note](06_documents/SIVP_20261117_UPDATE.md) identifies which result was replaced and why the earlier summaries are retained.

## Quick check without models or video

From the repository root, using Python 3:

```sh
python 07_scripts/verify_release.py
python 07_scripts/audit_recompute.py
python 07_scripts/audit_round2.py
python 07_scripts/audit_qwen25.py
```

These commands require only Python's standard library. The first checks file hashes. The next two retain the v1.0 audit logic and expected values; the final command checks the added Qwen2.5 records. They recompute defined checks from archived outputs, without training models, calling external services or using a GPU. Historical printed labels such as `4.1` refer to earlier manuscript sections, not current table numbers. See `AUDIT_REPORT.md` for the recorded check results and coverage.

## Contents

| Directory | Contents |
|---|---|
| `01_benchmark/` | Perception Test derived units and splits; Countix ID/split indexes |
| `02_predictions/` | Archived per-question model predictions; Countix files omit upstream labels and question text |
| `03_scores/` | Archived answer-score shards |
| `04_summaries/` | Aggregate results, including historical runs identified in the manifest |
| `05_figures/` | Earlier v1.0 PNGs; current main Figures 1–4 in `sivp_20261117/` |
| `06_documents/` | Reproduction and release-scope notes |
| `07_scripts/` | Three prediction audits, integrity check and archived research scripts |
| `08_history/` | Clearly marked withdrawn analysis and historical preparation records |
| `09_qwen25_update/` | New Qwen2.5 baseline and adapted-run records; English provenance note |

This repository does not include manuscript drafts, cover letters, internal review memos, raw video archives, extracted-frame directories, weights or hidden-state caches. The earlier Figure 6 is an attributed frame montage, as explained in `DATA_LICENSE.md`.

## Reproduction scope

The three prediction audits are ready to run from this repository. Other research scripts retain the original experiment paths and may require source annotations, sampled frames, model assets and the training environment. In particular, `canonical_eval_v2.py` needs the original frame eligibility checks and writes its output to its configured run directory; it is not one of the quick-check commands above. These files support checking archived outputs, not reproducing all training, inference or confidence intervals.

For Countix, this release provides IDs and this project's predictions but excludes upstream labels and question text. Obtain those from the cited source and follow `06_documents/COUNTIX_REPRODUCTION.md`. The reported target-refitted result uses `countix_refit_seed0.jsonl` through `seed4.jsonl`; earlier runs are not substitutes.

## Licensing and sources

Project software: MIT (`LICENSE`). Perception Test derived material and this project's outputs: CC BY 4.0, with the scope and third-party attribution in `DATA_LICENSE.md`. Countix upstream materials are not relicensed or included.

- Perception Test: https://github.com/google-deepmind/perception_test
- Countix / RepNet: https://sites.google.com/view/repnet
- Countix-QA distribution: https://huggingface.co/datasets/kekeYeah/Countix-QA

## Citation and version

The [v1.0 Zenodo archive](https://doi.org/10.5281/zenodo.24187160) retains the original title, **Supervision and Readout for Action-Conditioned Counting in Natural Videos**. `CITATION.cff` describes that archive and its DOI. **The v1.0 DOI does not include the later Qwen2.5 update or the new SIVP figures.**

When using the later material, also cite this GitHub repository at the full commit SHA you used. The 17 November update is a working-tree update; no new release version or DOI is assigned here.

Contact: Quan Ji, joker4@tjdz.edu.cn
