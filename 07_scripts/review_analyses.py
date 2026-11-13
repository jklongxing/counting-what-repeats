"""Reviewer-driven analyses.

1. Evidence density: does per-question accuracy depend on how many sampled slots fall
   inside the annotated events? If yes, the "evidence" explanation gains direct support.
2. Per-action mode baseline: the strongest constant predictor that uses the action name.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

RUNS = Path("/home/jiquan/video-state-evidence/repetition_readout_v1/runs")
UNITS = Path("/home/jiquan/video-state-evidence/repetition_readout_v1/p0_count_capacity/benchmark3_units.jsonl")


def bucket(c):
    return min(int(c), 4)


def load(path):
    return {r["unit_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())}


def support(unit):
    return sum(int(e[2]) for e in unit.get("events", []))


def main():
    units = {u["unit_id"]: u for u in (json.loads(l) for l in UNITS.read_text(encoding="utf-8").splitlines())}
    splits = {v: None for v in []}
    report = {}

    for tag, name in (("v3_test_base_32f", "zero-shot"), ("probe3_lora_test", "readout head")):
        preds = load(RUNS / f"{tag}.jsonl")
        rows = []
        for uid, row in preds.items():
            unit = units.get(uid)
            if unit is None:
                continue
            rows.append((support(unit), int(row["count"]), int(row["pred"])))
        buckets = defaultdict(lambda: [0, 0])
        for nsup, count, pred in rows:
            key = min(nsup, 4)
            buckets[key][1] += 1
            buckets[key][0] += int(bucket(pred) == bucket(count))
        report[name] = {str(k): dict(n=v[1], acc=round(v[0] / v[1], 4))
                        for k, v in sorted(buckets.items())}

    # per-action mode baseline (uses the training split only)
    train = [u for u in units.values() if u["split"] == "train"]
    test = [u for u in units.values() if u["split"] == "test"]
    mode = {}
    for action in {u["action"] for u in train}:
        counter = Counter(u["count"] for u in train if u["action"] == action)
        mode[action] = counter.most_common(1)[0][0] if counter else 1
    hits = sum(int(mode[u["action"]] == u["count"]) for u in test if u["action"] in mode)
    total = sum(1 for u in test if u["action"] in mode)
    report["per_action_mode_baseline"] = dict(n=total, accuracy=round(hits / total, 4),
                                               note="predicts the most common count per action from train")

    (RUNS / "review_analyses.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
