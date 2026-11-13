"""Consistent rescoring of the human-written questions and of the ordering analysis.

1. The three configurations must be scored with the same rule. The readout can only emit
   1-4, so raw exact match caps it; we therefore report four-way bucket accuracy for all
   three, and raw exact for the two text models with the readout's ceiling stated.
2. The ordering metric compared raw counts against predictions, so pairs whose true counts
   exceed 4 were mechanically lost for the readout. We restrict ordering to pairs with both
   true counts <= 4 and report the tie-aware variant as well.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"
UNITS = W / "p0_count_capacity" / "benchmark3_units.jsonl"
MC = W / "p0_count_capacity" / "mc_units.jsonl"
FRAMES = W / "data" / "frames"
OUT = RUNS / "rescore_mc_and_ordering.json"


def bucket(c):
    return min(int(c), 4)


def load(tag):
    path = RUNS / f"{tag}.jsonl"
    if not path.exists():
        return None
    return {r["unit_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())}


def pred_of(preds, uid):
    row = preds.get(uid)
    if not row:
        return None
    value = row.get("pred")
    return value if isinstance(value, int) else None


def mc_consistent():
    units = [json.loads(l) for l in MC.read_text(encoding="utf-8").splitlines()]
    readout = json.loads((RUNS / "mc_readout.json").read_text(encoding="utf-8"))
    methods = {"zero-shot": load("mc_qwen3vl_32f"), "LoRA": load("mc_lora")}
    out = {"n": len(units), "count_range": [min(u["count"] for u in units), max(u["count"] for u in units)],
           "n_above_4": sum(1 for u in units if u["count"] > 4)}
    for name, preds in methods.items():
        exact = buck = parsed = 0
        for u in units:
            p = pred_of(preds, u["unit_id"])
            if p is None:
                continue
            parsed += 1
            exact += int(p == u["count"])
            buck += int(bucket(p) == bucket(u["count"]))
        out[name] = dict(parsed=parsed, exact=round(exact / len(units), 4),
                         bucket=round(buck / len(units), 4))
    out["readout"] = dict(note="classifier emits 1-4 only",
                          bucket=readout["mean"], bucket_sd=readout["sd"],
                          bucket_seeds={k: v for k, v in readout.items() if k.startswith("seed")})
    return out


def ordering():
    units = [json.loads(l) for l in UNITS.read_text(encoding="utf-8").splitlines()]
    units = [u for u in units if (FRAMES / u["video_id"]).exists()
             and len(list((FRAMES / u["video_id"]).glob("*.jpg"))) >= 32]
    test = [u for u in units if u["split"] == "test"]
    by_video = defaultdict(list)
    for u in test:
        by_video[u["video_id"]].append(u)
    pairs = []
    for video, rows in by_video.items():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a["action"] != b["action"] and a["count"] != b["count"]:
                    pairs.append((a, b))

    lora, head, base = load("v3_test_lora_32f"), load("probe3_lora_test"), load("v3_test_base_32f")
    out = {"pairs_all": len(pairs), "pairs_le4": sum(1 for a, b in pairs if max(a["count"], b["count"]) <= 4)}
    for name, preds in (("zero-shot", base), ("LoRA", lora), ("readout", head)):
        stats = {}
        for label, subset in (("all", pairs),
                              ("both_le_4", [(a, b) for a, b in pairs if max(a["count"], b["count"]) <= 4])):
            strict = tie_ok = n = 0
            for a, b in subset:
                pa, pb = pred_of(preds, a["unit_id"]), pred_of(preds, b["unit_id"])
                if pa is None or pb is None:
                    continue
                n += 1
                up = a["count"] < b["count"]
                strict += int(pa < pb) if up else int(pa > pb)
                tie_ok += int(pa <= pb) if up else int(pa >= pb)
            stats[label] = dict(n=n, strict=round(strict / max(n, 1), 4), tie_aware=round(tie_ok / max(n, 1), 4))
        out[name] = stats
    return out


def main():
    report = {"mc": mc_consistent(), "ordering": ordering()}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
