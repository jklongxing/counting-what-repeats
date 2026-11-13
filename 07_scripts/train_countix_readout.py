"""Refit the Countix readout excluding every cut of any source video seen in validation.

Clip names are <source>_cut<N>. Excluding only the identical clip leaves other cuts of the same
source in the fitting set, so we drop the whole source for any source that appears in the
evaluation set.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"
CUT = re.compile(r"_cut\d+$")


def b4(c):
    return min(int(c), 4)


def load(name):
    blob = torch.load(RUNS / name, map_location="cpu", weights_only=False)
    rows = blob["units"]
    return rows, torch.stack([r["hidden"] for r in rows]).numpy()


def source(video_id):
    return CUT.sub("", video_id)


def main():
    tr_rows, Xtr = load("h3_countix_train_Qwen3-VL-8B-Instruct_32f.pt")
    va_rows, Xva = load("h3_lora_countix_Qwen3-VL-8B-Instruct_32f.pt")
    val_sources = {source(r["video_id"]) for r in va_rows}
    keep = np.array([source(r["video_id"]) not in val_sources for r in tr_rows])
    dropped = [r["video_id"] for r, k in zip(tr_rows, keep) if not k]
    print("val sources:", len(val_sources), "dropped train clips:", len(dropped), dropped)

    Xtr, tr_rows = Xtr[keep], [r for r, k in zip(tr_rows, keep) if k]
    ytr = np.array([b4(r["count"]) for r in tr_rows])
    yva = np.array([b4(r["count"]) for r in va_rows])
    print("train units now:", len(ytr), Counter(ytr.tolist()).most_common())

    out = {"train": len(ytr), "val": len(yva),
           "val_sources": len(val_sources), "dropped_train_clips": dropped,
           "train_sources": len({source(r["video_id"]) for r in tr_rows}),
           "source_overlap": 0,
           "train_dist": {str(k): v for k, v in sorted(Counter(ytr.tolist()).items())},
           "val_dist": {str(k): v for k, v in sorted(Counter(yva.tolist()).items())},
           "constant2_baseline": round(float((yva == 2).mean()), 4)}
    accs = []
    for seed in range(5):
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=400,
                                          early_stopping=True, n_iter_no_change=15, random_state=seed))
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
        acc = float((pred == yva).mean())
        accs.append(acc)
        out[f"seed{seed}"] = round(acc, 4)
        with (RUNS / f"countix_refit_seed{seed}.jsonl").open("w", encoding="utf-8") as handle:
            for row, q in zip(va_rows, pred):
                handle.write(json.dumps(dict(unit_id=row["unit_id"], video_id=row["video_id"],
                                             count=int(row["count"]), pred=int(q)), ensure_ascii=False) + "\n")
    out["mean"] = round(float(np.mean(accs)), 4)
    out["sd"] = round(float(np.std(accs)), 4)
    (RUNS / "countix_refit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
