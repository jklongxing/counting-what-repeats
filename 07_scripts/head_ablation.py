"""Head capacity ablation and Wilson confidence intervals.

1. Linear (multinomial logistic) head versus the MLP head on the same features, to show
   that the gain does not depend on a flexible classifier.
2. Wilson 95% intervals for the headline accuracies, so readers can judge differences
   against sampling noise.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUNS = Path("/home/jiquan/video-state-evidence/repetition_readout_v1/runs")


def bucket(c):
    return min(int(c), 4)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def load(tag):
    blob = torch.load(RUNS / f"{tag}_Qwen3-VL-8B-Instruct_32f.pt", map_location="cpu", weights_only=False)
    rows = blob["units"]
    return rows, torch.stack([r["hidden"] for r in rows]).numpy()


def main():
    rows, X = load("h3_lora")
    split = np.array([r["split"] for r in rows])
    y = np.array([bucket(r["count"]) for r in rows])
    tr, te = split == "train", split == "test"

    out = {}
    for name, clf in (
        ("linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))),
        ("mlp", make_pipeline(StandardScaler(), __import__("sklearn.neural_network", fromlist=["MLPClassifier"]).MLPClassifier(
            hidden_layer_sizes=(256,), alpha=1e-3, max_iter=400, early_stopping=True,
            n_iter_no_change=15, random_state=0))),
    ):
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        gt = y[te]
        hit = pred == gt
        out[name] = dict(
            overall=round(float(hit.mean()), 4),
            overall_ci=wilson(int(hit.sum()), len(hit)),
            once=round(float((pred[gt == 1] == 1).mean()), 4),
            repeated=round(float(hit[gt >= 2].mean()), 4),
            repeated_ci=wilson(int(hit[gt >= 2].sum()), int((gt >= 2).sum())))

    # Wilson intervals for the text configurations
    for tag, key in (("v3_test_lora_32f", "lora_text"), ("v3_test_base_32f", "zero_shot")):
        rows_t = [json.loads(l) for l in (RUNS / f"{tag}.jsonl").read_text(encoding="utf-8").splitlines()]
        rows_t = [r for r in rows_t if isinstance(r.get("pred"), int)]
        rep = [r for r in rows_t if r["count"] >= 2]
        k = sum(int(bucket(r["pred"]) == bucket(r["count"])) for r in rep)
        out[key] = dict(repeated=round(k / len(rep), 4), repeated_ci=wilson(k, len(rep)))

    (RUNS / "head_ablation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
