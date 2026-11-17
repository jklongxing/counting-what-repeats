# Qwen2.5-VL-7B: training records and results

Article: Overall Accuracy Gains Mask Repetition Errors in Supervised Video Counting

Journal: Signal, Image and Video Processing

Author and corresponding author: Quan Ji

Affiliation: Tianjin Electronic Information College, Tianjin, China

Email: joker4@tjdz.edu.cn

This note documents the Qwen2.5-VL-7B results in Table 1 and Section 4.1 of the main manuscript. It accompanies the commands, saved configurations and per-question predictions in `ESM_2_Qwen25_Provenance.zip`, referenced from Online Resource 1.

An earlier archived result (73.0% overall accuracy, 9.4% repeated-action accuracy and a 96.7% answer-one share) could not be reproduced from the recorded recipe. We replaced it with two newly trained runs, repro1 and repro2, for which the commands and saved training arguments are included. Table 1 reports repro1; Section 4.1 also reports repro2.

Both runs were evaluated on the same 577 test questions using 32 frames, a 512-squared pixel budget, greedy decoding and at most 12 new tokens. Accuracy uses the four count buckets 1, 2, 3 and 4+. The test set contains 406 single-occurrence and 171 repeated-action questions.

| Run | Overall accuracy (%) | Single-occurrence accuracy (%) | Repeated-action accuracy (%) | Answer-one share (%) |
|---|---:|---:|---:|---:|
| Zero-shot baseline | 41.4 | 49.5 | 22.2 | 49.6 |
| repro1 | 78.3 | 97.8 | 32.2 | 84.9 |
| repro2 | 78.0 | 94.1 | 39.8 | 79.7 |

The re-evaluated baseline matches the archived predictions item for item. Both adapted runs improve repeated-action accuracy, with larger gains on single-occurrence questions. Their answer-one shares also increase relative to the baseline.

The two runs have the same recorded hyperparameters and seed argument: LoRA rank 16, alpha 32, dropout 0.05, q/k/v/o/gate/up/down projection targets, one epoch, a learning rate of 1e-4, gradient accumulation of 8, weight decay of 0.01 and `seed=0`. The training command sets `--max-pixels 448`. The commands differ in GPU ID (2 versus 3) and output tag. Repeated-action accuracy and the answer-one share vary between runs, but these experiments do not isolate the source of that variation.

The records below support checking the reported settings and scores. The training script, model weights and adapter weights are not included in this supplement.

| Files | Contents |
|---|---|
| `repro1_command.txt`, `repro2_command.txt` | Recorded training commands |
| `repro1_training_args.json`, `repro2_training_args.json` | Saved training arguments and LoRA configurations |
| `base_test_summary.json`, `base_test_predictions.jsonl` | Baseline summary and per-question predictions |
| `repro1_test_summary.json`, `repro1_test_predictions.jsonl` | First run summary and per-question predictions |
| `repro2_test_summary.json`, `repro2_test_predictions.jsonl` | Second run summary and per-question predictions |
| `repro_check_60.txt`, `repro60_q25v3_records.jsonl`, `repro60_records.log` | Historical 60-question check of the earlier archived adapter |

The historical 60-question check concerns the earlier adapter, `lora_q25_v3`. It does not verify repro1 or repro2 and is retained only to document the earlier record; the manuscript uses the new runs.
