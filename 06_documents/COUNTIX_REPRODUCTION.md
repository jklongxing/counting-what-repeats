# Countix source labels and released predictions

This release does not redistribute the upstream Countix-QA ground-truth annotations or question text. The public upstream page did not specify a dataset license when checked on 13 November 2026. The Perception Test license must not be attributed to Countix.

Source distribution: https://huggingface.co/datasets/kekeYeah/Countix-QA
Original dataset/project: https://sites.google.com/view/repnet

`01_benchmark/countix_units.jsonl` and `countix_train_units.jsonl` retain `unit_id`, `video_id` and `split` as join indexes. `02_predictions/countix*.jsonl` retain this project's numerical predictions with the corresponding source identifiers and optional frame budget. Source labels, question/answer fields and unused extra fields were removed from these release copies; the local source archive is preserved. Numerical predictions and their row order are unchanged.

To recompute the Countix accuracies, obtain source annotations under their applicable terms, recover the count for each released unit, and create a local CSV with columns `unit_id,count`. Match IDs explicitly; do not infer a count from a model answer. Then run:

```sh
python 07_scripts/recompute_countix.py --labels /path/to/locally_obtained_labels.csv
```

The helper checks a one-to-one label lookup for every released prediction, requires each final refit run to have 476 unique units, pools counts at 4, and reports each file's four-way accuracy. It does not download upstream data. The helper was checked using the existing local archived labels: the two text-model rows and five refit rows reproduce. A fresh upstream download and its identifier mapping were not tested in this preparation session. The non-refitted readout row (34.0%) is available as an archived aggregate in `countix_readout.json`, not as a per-question file scored by this helper. The archived five-seed target-refitted result is 56.47% with standard deviation 0.95 percentage points; the manuscript reports 56.5 +/- 1.0. The aggregate `04_summaries/countix_refit.json` is preserved unchanged.

The original `build_countix_frames.py` and `build_countix_train.py` are included as historical construction code and require the original experiment environment. They are not automatically run by any release check.
