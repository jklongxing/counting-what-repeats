# Counting what repeats

Evaluation artifacts for **Supervision and Readout for Action-Conditioned Counting in Natural Videos**, by **Quan Ji**, Tianjin Electronic Information College.

This repository contains the derived benchmark, archived model predictions, answer scores, aggregate results, final manuscript figures and analysis scripts. The main benchmark has 3,710 evaluable questions over 1,563 videos and 59 action labels. It separates single-occurrence from repeated-action questions. On the standard adapter, the four-way readout raises repeated-action accuracy from 43.9% to 56.7%, with a 3.7-point decrease on single-occurrence questions. See `MANIFEST.md` for the result-to-file map and `REPRODUCIBILITY.md` for what can be recomputed from this release.

## Quick check without models or video

From the repository root, using Python 3:

```sh
python 07_scripts/verify_release.py
python 07_scripts/audit_recompute.py
python 07_scripts/audit_round2.py
```

These commands require only Python's standard library. The first checks file hashes; the two audits recompute their defined checks from archived predictions. They do not train models, call external services or use a GPU. Their historical printed labels such as `4.1` refer to manuscript section numbers, not current table numbers. See `AUDIT_REPORT.md` for the actual release-check results and coverage.

## Contents

| Directory | Contents |
|---|---|
| `01_benchmark/` | Perception Test derived units and splits; Countix ID/split indexes |
| `02_predictions/` | Archived per-question model predictions; Countix files omit upstream labels and question text |
| `03_scores/` | Archived answer-score shards |
| `04_summaries/` | Aggregate results, including historical runs identified in the manifest |
| `05_figures/` | Figures 1–8 and graphical abstract synchronized with manuscript v11 |
| `06_documents/` | Reproduction and release-scope notes |
| `07_scripts/` | Two runnable prediction audits, integrity check and archived research scripts |
| `08_history/` | Clearly marked withdrawn analysis and historical preparation records |

The release contains no old manuscript drafts, cover letters, internal review memos, raw video archives, extracted-frame directories, weights or hidden-state caches. Figure 6 is an attributed frame montage, as explained in `DATA_LICENSE.md`.

## Reproduction scope

The two prediction audits are ready to run from this repository. Other research scripts retain the original experiment paths and may require source annotations, sampled frames, model assets and the training environment. In particular, `canonical_eval_v2.py` needs the original frame eligibility checks and writes its output to its configured run directory; it is not one of the quick-check commands above. This release does not claim to reproduce training or all confidence intervals from only these files.

For Countix, this release provides IDs and this project's predictions but excludes upstream labels and question text. Obtain those from the cited source and follow `06_documents/COUNTIX_REPRODUCTION.md`. The reported target-refitted result uses `countix_refit_seed0.jsonl` through `seed4.jsonl`; earlier runs are not substitutes.

## Licensing and sources

Project software: MIT (`LICENSE`). Perception Test derived material and this project's outputs: CC BY 4.0, with the scope and third-party attribution in `DATA_LICENSE.md`. Countix upstream materials are not relicensed or included.

- Perception Test: https://github.com/google-deepmind/perception_test
- Countix / RepNet: https://sites.google.com/view/repnet
- Countix-QA distribution: https://huggingface.co/datasets/kekeYeah/Countix-QA

## Citation and version

Use `CITATION.cff`. The initial release is intended for tag `v1.0`. A version-specific DOI will be added after Zenodo has archived that release. No DOI or journal acceptance is claimed before it exists.

Contact: Quan Ji, joker4@tjdz.edu.cn
