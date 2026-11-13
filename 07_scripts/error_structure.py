"""Error-structure analyses requested by a strict review.

1. Per-action accuracy (zero-shot, LoRA text, readout head).
2. Accuracy by true count (1, 2, 3, 4+).
3. Rank correlation between predicted and true counts.
4. McNemar tests against the per-action prior.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import spearmanr

RUNS = Path("/home/jiquan/video-state-evidence/repetition_readout_v1/runs")
UNITS = Path("/home/jiquan/video-state-evidence/repetition_readout_v1/p0_count_capacity/benchmark3_units.jsonl")


def bucket(c):
    return min(int(c), 4)


def load(tag):
    return {r["unit_id"]: r for r in (json.loads(l) for l in (RUNS / f"{tag}.jsonl").read_text(encoding="utf-8").splitlines())}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n))


def main():
    units = {u["unit_id"]: u for u in (json.loads(l) for l in UNITS.read_text(encoding="utf-8").splitlines())}
    train = [u for u in units.values() if u["split"] == "train"]
    mode = {}
    for action in {u["action"] for u in train}:
        counter = Counter(u["count"] for u in train if u["action"] == action)
        mode[action] = counter.most_common(1)[0][0]

    methods = {"zero-shot": load("v3_test_base_32f"),
               "LoRA": load("v3_test_lora_32f"),
               "head": load("probe3_lora_test")}
    ids = [i for i in methods["zero-shot"] if i in units and units[i]["split"] == "test"]

    report = {"per_action": {}, "by_count": {}, "correlation": {}, "mcnemar_vs_prior": {}}
    actions = sorted({units[i]["action"] for i in ids})

    for name, preds in methods.items():
        per_action = {}
        for action in actions:
            subset = [i for i in ids if units[i]["action"] == action]
            hits = sum(int(bucket(preds[i]["pred"]) == bucket(units[i]["count"])) for i in subset if i in preds)
            per_action[action] = [len(subset), round(hits / len(subset), 3)]
        report["per_action"][name] = per_action

        by_count = {}
        for count in (1, 2, 3, 4):
            subset = [i for i in ids if min(units[i]["count"], 4) == count]
            hits = sum(int(bucket(preds[i]["pred"]) == count) for i in subset if i in preds)
            by_count[str(count)] = [len(subset), round(hits / max(len(subset), 1), 3)]
        report["by_count"][name] = by_count

        pairs = [(preds[i]["pred"], units[i]["count"]) for i in ids if i in preds]
        rho, p = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
        report["correlation"][name] = dict(spearman=round(float(rho), 3), p=float(p))

        # McNemar against the prior
        b = c = 0
        for i in ids:
            if i not in preds:
                continue
            prior_ok = bucket(mode[units[i]["action"]]) == bucket(units[i]["count"])
            model_ok = bucket(preds[i]["pred"]) == bucket(units[i]["count"])
            if model_ok and not prior_ok:
                b += 1
            elif prior_ok and not model_ok:
                c += 1
        report["mcnemar_vs_prior"][name] = dict(model_only=b, prior_only=c, p=round(mcnemar(b, c), 6))

    (RUNS / "error_structure.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
