"""Output calibration baseline: a learned per-class bias on the answer scores.

For a question with score vector s (one entry per digit 1..8) the calibrated answer is
argmax_c (s_c + alpha * b_c). The bias b is fitted on the training split, the strength alpha
is chosen on dev, and both are frozen before evaluating test and VAL. This is the standard
output-calibration baseline: it never touches the representation, only the final scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"
OUT = RUNS / "output_calibration.json"


def bucket(c):
    return min(int(c), 4)


def load_scores(pattern="scores_lora_all_s*.jsonl"):
    rows = {}
    for path in sorted(RUNS.glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[row["unit_id"]] = row
    return rows


def accuracy(scores, counts, bias, alpha):
    pred = np.argmax(scores + alpha * bias, axis=1) + 1
    hit4 = (np.minimum(pred, 4) == np.minimum(counts, 4))
    once = counts == 1
    rep = counts >= 2
    return dict(overall=float(hit4.mean()),
                single=float(hit4[once].mean()) if once.any() else None,
                repeated=float(hit4[rep].mean()) if rep.any() else None)


def fit_bias(scores, counts, passes=3, grid=np.arange(-2.0, 2.01, 0.25)):
    bias = np.zeros(scores.shape[1])
    best = accuracy(scores, counts, bias, 1.0)["overall"]
    for _ in range(passes):
        improved = False
        for c in range(scores.shape[1]):
            original = bias[c]
            for value in grid:
                bias[c] = value
                score = accuracy(scores, counts, bias, 1.0)["overall"]
                if score > best + 1e-9:
                    best, improved = score, True
                    original = value
            bias[c] = original
        if not improved:
            break
    return bias, best


def main():
    rows = load_scores()
    by_split = {"train": [], "dev": [], "test": [], "val": []}
    for row in rows.values():
        if row["split"] in by_split:
            by_split[row["split"]].append(row)

    def pack(items):
        return (np.array([r["scores"] for r in items], dtype=np.float64),
                np.array([int(r["count"]) for r in items]))

    xs, ys = pack(by_split["train"])
    bias, train_acc = fit_bias(xs, ys)
    xd, yd = pack(by_split["dev"])

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    dev_curve = {str(a): accuracy(xd, yd, bias, a) for a in alphas}
    best_alpha = max(alphas, key=lambda a: dev_curve[str(a)]["overall"])

    report = dict(train_units=len(ys), bias=[round(float(v), 3) for v in bias],
                  best_on_train=round(train_acc, 4),
                  alphas=dev_curve, chosen_alpha=best_alpha,
                  dev=dev_curve[str(best_alpha)], test={}, val={})
    for split in ("test", "val"):
        if not by_split[split]:
            continue
        x, y = pack(by_split[split])
        report[split] = {"text": accuracy(x, y, bias, 0.0),
                         "calibrated": accuracy(x, y, bias, best_alpha)}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
