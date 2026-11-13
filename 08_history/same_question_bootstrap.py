"""Two-way video-clustered bootstrap for the same-question pairs.

Each pair joins two different videos. A unit appears in at most one pair, but a video can
appear in several pairs through different actions, so resampling pairs independently
understates the interval. We use the two-way cluster bootstrap: draw a sample of the
single-occurrence-side videos and of the repeated-side videos independently, weight each pair
by w = 1[a in A] + 1[b in B] - 1[a in A] * 1[b in B], and recompute the difference.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

B = Path(__file__).resolve().parent / "DATA_BUNDLE_20261111"
PRED = B / "02_predictions"
UNITS = [json.loads(l) for l in (B / "01_benchmark" / "evaluable_units.jsonl").read_text(encoding="utf-8").splitlines()]
OUT = B / "04_summaries" / "same_question_bootstrap.json"


def b4(c):
    return min(int(c), 4)


def load(tag):
    return {json.loads(l)["unit_id"]: json.loads(l).get("pred") for l in (PRED / f"{tag}.jsonl").read_text(encoding="utf-8").splitlines()}


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

    lora, head = load("v3_test_lora_32f"), load("probe3_lora_test")

    def both(p, a, b):
        pa, pb = p.get(a["unit_id"]), p.get(b["unit_id"])
        return isinstance(pa, int) and isinstance(pb, int) and b4(pa) == b4(a["count"]) and b4(pb) == b4(b["count"])

    va = [a["video_id"] for a, _ in pairs]
    vb = [b["video_id"] for _, b in pairs]
    set_a, set_b = sorted(set(va)), sorted(set(vb))
    idx_a = {v: i for i, v in enumerate(set_a)}
    idx_b = {v: i for i, v in enumerate(set_b)}
    pa_idx = [idx_a[v] for v in va]
    pb_idx = [idx_b[v] for v in vb]

    d = [int(both(head, a, b)) - int(both(lora, a, b)) for a, b in pairs]
    point = sum(d) / len(d)

    rng = random.Random(0)
    draws = []
    na, nb = len(set_a), len(set_b)
    for _ in range(2000):
        sa = [0] * na
        sb = [0] * nb
        for _ in range(na):
            sa[rng.randrange(na)] += 1
        for _ in range(nb):
            sb[rng.randrange(nb)] += 1
        num = den = 0.0
        for k in range(len(pairs)):
            w = sa[pa_idx[k]] + sb[pb_idx[k]] - sa[pa_idx[k]] * sb[pb_idx[k]]
            if w:
                num += w * d[k]
                den += w
        draws.append(num / den if den else 0.0)
    draws.sort()
    lo, hi = draws[50], draws[1949]
    below = sum(1 for x in draws if x <= 0)
    p = 2 * min(below, 2000 - below) / 2000
    p_text = f"< {2/2000:.4f}" if p == 0 else f"{p:.4f}"

    report = dict(pairs=len(pairs), units=2 * len(pairs), videos_total=len(set(va) | set(vb)),
                  videos_once_side=na, videos_repeated_side=nb,
                  delta_pp=round(point * 100, 2),
                  ci_pp=[round(lo * 100, 1), round(hi * 100, 1)], p=p_text,
                  note="two-way video-cluster bootstrap, 2000 draws")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
