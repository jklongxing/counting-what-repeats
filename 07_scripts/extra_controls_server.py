"""Server-side run of the two extra controls (see extra_controls.py).

Adds the paired test for the same-question pairs and the score-space readout, using the
saved eight-digit scores.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"
UNITS = [json.loads(l) for l in (RUNS / "evaluable_units.jsonl").read_text(encoding="utf-8").splitlines()]
VAL = [json.loads(l) for l in (W / "p0_count_capacity" / "val_units.jsonl").read_text(encoding="utf-8").splitlines()]
OUT = RUNS / "extra_controls.json"


def b4(c):
    return min(int(c), 4)


def load(tag):
    path = RUNS / f"{tag}.jsonl"
    if not path.exists():
        return None
    return {json.loads(l)["unit_id"]: json.loads(l).get("pred") for l in path.read_text(encoding="utf-8").splitlines()}


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def main():
    test = [u for u in UNITS if u["split"] == "test"]
    by_action = defaultdict(lambda: {"once": [], "repeated": []})
    for u in test:
        by_action[u["action"]]["once" if u["count"] == 1 else "repeated"].append(u)
    pairs = []
    for action in sorted(by_action):
        once = sorted(by_action[action]["once"], key=lambda u: u["unit_id"])
        rep = sorted(by_action[action]["repeated"], key=lambda u: u["unit_id"])
        for a, b in zip(once, rep):
            if b4(a["count"]) != b4(b["count"]):
                pairs.append((a, b))

    lora, head, base = load("v3_test_lora_32f"), load("probe3_lora_test"), load("v3_test_base_32f")

    def both(p, a, b):
        pa, pb = p.get(a["unit_id"]), p.get(b["unit_id"])
        return isinstance(pa, int) and isinstance(pb, int) and b4(pa) == b4(a["count"]) and b4(pb) == b4(b["count"])

    report = {"same_question": {"pairs": len(pairs), "actions": len({a["action"] for a, _ in pairs})}}
    for name, p in (("zero-shot", base), ("LoRA", lora), ("readout", head)):
        hit = sum(1 for a, b in pairs if both(p, a, b))
        report["same_question"][name] = dict(both=hit, rate=round(hit / len(pairs), 4), ci=wilson(hit, len(pairs)))

    # paired bootstrap over pairs (each unit appears at most once)
    def delta(f, g):
        vals = [int(f(a, b)) - int(g(a, b)) for a, b in pairs]
        point = sum(vals) / len(vals)
        rng = random.Random(0)
        draws = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals) for _ in range(2000))
        below = sum(1 for d in draws if d <= 0)
        p = 2 * min(below, 2000 - below) / 2000
        return dict(delta=round(point, 4), ci=[round(draws[50], 4), round(draws[1949], 4)],
                    p=f"< {2/2000:.4f}" if p == 0 else f"{p:.4f}")
    report["same_question"]["tests"] = {
        "readout_minus_lora": delta(lambda a, b: both(head, a, b), lambda a, b: both(lora, a, b)),
        "readout_minus_zeroshot": delta(lambda a, b: both(head, a, b), lambda a, b: both(base, a, b)),
    }

    # ---- score-space readout ----
    scores = {}
    for shard in sorted(RUNS.glob("scores_lora_all_s*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            scores[row["unit_id"]] = row

    def pack(split):
        ids = [u["unit_id"] for u in UNITS + VAL if u["split"] == split and u["unit_id"] in scores]
        X = np.array([scores[i]["scores"] for i in ids], dtype=np.float64)
        y = np.array([b4(scores[i]["count"]) for i in ids])
        return ids, X, y

    ids_tr, X_tr, y_tr = pack("train")
    out = {"score_head": {"train": int(len(y_tr))}}
    for split, tag in (("test", "probe3_lora_test"), ("val", "probe3_lora_val")):
        ids, X, y = pack(split)
        accs = []
        for seed in range(3):
            clf = make_pipeline(StandardScaler(),
                                MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=400,
                                              early_stopping=True, n_iter_no_change=15, random_state=seed))
            clf.fit(X_tr, y_tr)
            accs.append(float((clf.predict(X) == y).mean()))
        ref = load(tag)

        def breakdown(getter):
            once = [(getter(i), yy) for i, yy in zip(ids, y) if yy == 1]
            rep = [(getter(i), yy) for i, yy in zip(ids, y) if yy >= 2]
            overall = sum(1 for i, yy in zip(ids, y) if getter(i) == yy) / len(ids)
            return dict(overall=round(overall, 4),
                        single=round(sum(1 for q, _ in once if q == 1) / max(len(once), 1), 4),
                        repeated=round(sum(1 for q, yy in rep if q == yy) / max(len(rep), 1), 4))

        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=400,
                                          early_stopping=True, n_iter_no_change=15, random_state=0))
        clf.fit(X_tr, y_tr)
        score_pred = clf.predict(X)
        score_map = {i: int(q) for i, q in zip(ids, score_pred)}
        with (RUNS / f"scorehead_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for i, q, yy in zip(ids, score_pred, y):
                handle.write(json.dumps(dict(unit_id=i, count=int(scores[i]["count"]),
                                             pred=int(q), split=split), ensure_ascii=False) + "\n")
        hidden_map = {i: (b4(ref[i]) if isinstance(ref.get(i), int) else -1) for i in ids}
        out["score_head"][split] = dict(n=int(len(y)),
                                        score_head=breakdown(lambda i: score_map[i]),
                                        hidden_readout=breakdown(lambda i: hidden_map[i]),
                                        score_head_sd=round(float(np.std(accs)), 4))
    report.update(out)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
