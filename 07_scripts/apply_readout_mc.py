"""Apply the main-benchmark readout to the human-written multiple-choice questions.

The readout is trained on the main benchmark's training split (same adapter, same hidden
state, same four-way task) and then applied unchanged to the 346 human-written questions.
Nothing is tuned on the human questions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"


def bucket(c):
    return min(int(c), 4)


def load(name):
    blob = torch.load(RUNS / name, map_location="cpu", weights_only=False)
    rows = blob["units"]
    return rows, torch.stack([r["hidden"] for r in rows]).numpy()


def main():
    rows, X = load("h3_lora_Qwen3-VL-8B-Instruct_32f.pt")
    split = np.array([r["split"] for r in rows])
    y = np.array([bucket(r["count"]) for r in rows])
    tr = split == "train"

    mc_rows, X_mc = load("h3_lora_mc_Qwen3-VL-8B-Instruct_32f.pt")
    y_mc = np.array([bucket(r["count"]) for r in mc_rows])

    out = {"n_mc": len(mc_rows)}
    for seed in range(5):
        clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3,
                                                            max_iter=400, early_stopping=True,
                                                            n_iter_no_change=15, random_state=seed))
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X_mc)
        hit = (pred == y_mc).mean()
        out[f"seed{seed}"] = round(float(hit), 4)
        if seed == 0:
            dist = {str(k): int((pred == k).sum()) for k in (1, 2, 3, 4)}
    values = [out[f"seed{s}"] for s in range(5)]
    out["mean"] = round(float(np.mean(values)), 4)
    out["sd"] = round(float(np.std(values)), 4)
    out["pred_dist_seed0"] = dist
    out["true_dist"] = {str(k): int((y_mc == k).sum()) for k in (1, 2, 3, 4)}
    (RUNS / "mc_readout.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
