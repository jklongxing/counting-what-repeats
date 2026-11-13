"""Readout head across adapter seeds.

For every adapter seed we train the four-way head on that adapter's own training
representations and evaluate on the matching test representations, so the comparison is
always seed-matched. Five head seeds per adapter give the spread.

Usage: python multiseed_probe.py --hidden "h3_lora,h3_seed7,h3_seed8" --labels "seed0,seed7,seed8"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "p0_count_capacity").exists() else HERE.parent
RUNS = ROOT / "runs"


def bucket(c):
    return min(int(c), 4)


def load(tag):
    path = RUNS / f"{tag}_Qwen3-VL-8B-Instruct_32f.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    rows = blob["units"]
    X = torch.stack([r["hidden"] for r in rows]).numpy()
    return rows, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--head-seeds", dest="head_seeds", type=int, default=5)
    ap.add_argument("--out", default="multiseed_probe.json")
    args = ap.parse_args()

    tags = [t.strip() for t in args.hidden.split(",")]
    labels = [t.strip() for t in args.labels.split(",")]
    report = []
    for tag, label in zip(tags, labels):
        rows, X = load(tag)
        split = np.array([r["split"] for r in rows])
        raw = np.array([r["count"] for r in rows])
        y = np.array([bucket(c) for c in raw])
        tr, te = split == "train", split == "test"
        line = {"adapter": label}
        hits = []
        for seed in range(args.head_seeds):
            clf = make_pipeline(StandardScaler(),
                                MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3,
                                              max_iter=400, early_stopping=True,
                                              n_iter_no_change=15, random_state=seed))
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            gt = y[te]
            hit = pred == gt
            hits.append((hit.mean(),
                         (pred[gt == 1] == 1).mean() if (gt == 1).any() else 0.0,
                         hit[gt >= 2].mean() if (gt >= 2).any() else 0.0))
        arr = np.array(hits)
        line.update(head_overall=round(float(arr[:, 0].mean()), 4),
                    head_overall_sd=round(float(arr[:, 0].std()), 4),
                    head_once=round(float(arr[:, 1].mean()), 4),
                    head_repeated=round(float(arr[:, 2].mean()), 4),
                    head_repeated_sd=round(float(arr[:, 2].std()), 4))
        report.append(line)
        print(json.dumps(line, ensure_ascii=False), flush=True)

    (RUNS / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

