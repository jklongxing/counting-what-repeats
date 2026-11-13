"""Apply the Perception-Test readout to the Countix clips (cross-dataset test)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"
OUT = RUNS / "countix_readout.json"


def b4(c):
    return min(int(c), 4)


def load(name):
    blob = torch.load(RUNS / name, map_location="cpu", weights_only=False)
    rows = blob["units"]
    return rows, torch.stack([r["hidden"] for r in rows]).numpy()


def main():
    rows, X = load("h3_lora_Qwen3-VL-8B-Instruct_32f.pt")
    split = np.array([r["split"] for r in rows])
    y = np.array([b4(r["count"]) for r in rows])
    tr = split == "train"

    cx, Xc = load("h3_lora_countix_Qwen3-VL-8B-Instruct_32f.pt")
    yc = np.array([b4(r["count"]) for r in cx])
    raw = np.array([int(r["count"]) for r in cx])

    out = {"n": len(cx), "true_dist": dict(Counter(int(c) for c in raw))}
    preds = {}
    for seed in range(3):
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=400,
                                          early_stopping=True, n_iter_no_change=15, random_state=seed))
        clf.fit(X[tr], y[tr])
        p = clf.predict(Xc)
        preds[seed] = p
        out[f"seed{seed}"] = dict(overall=round(float((p == yc).mean()), 4),
                                  exact_raw=round(float((np.minimum(p, 4) == np.minimum(raw, 4)).mean()), 4))
    best = preds[0]
    out["per_count"] = {str(c): round(float((best[raw == c] == yc[raw == c]).mean()), 4)
                        for c in sorted(set(int(v) for v in raw))}
    out["pred_dist_seed0"] = dict(Counter(int(v) for v in best))
    mean_acc = np.mean([out[f"seed{s}"]["overall"] for s in range(3)])
    out["mean"] = round(float(mean_acc), 4)
    (RUNS / "countix_readout.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
