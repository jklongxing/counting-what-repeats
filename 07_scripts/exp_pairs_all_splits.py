"""Same-video pairs across every held-out split (dev, test, VAL) and pooled.

Dev and test predictions exist for zero-shot, LoRA and the readout, so the joint test can
be run on both, giving a larger sample than the test split alone. VAL is reported for
completeness but is small.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"
UNITS = W / "p0_count_capacity" / "benchmark3_units.jsonl"
VAL = W / "p0_count_capacity" / "val_units.jsonl"
OUT = RUNS / "same_video_all_splits.json"

SPLITS = {
    "test": dict(units=UNITS, split="test",
                 preds=dict(base="v3_test_base_32f", lora="v3_test_lora_32f", head="probe3_lora_test")),
    "dev": dict(units=UNITS, split="dev",
                preds=dict(base="v3_dev_base_32f", lora="v3_dev_lora_32f", head="probe3_lora_dev")),
    "val": dict(units=VAL, split="val",
                preds=dict(base="val_base_32f", lora="val_lora_32f", head="probe3_lora_val")),
}


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


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def build_pairs(units, split):
    by_video = defaultdict(list)
    for u in units:
        if u["split"] == split:
            by_video[u["video_id"]].append(u)
    pairs = defaultdict(list)
    for video, rows in by_video.items():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a["action"] != b["action"] and a["count"] != b["count"]:
                    pairs[video].append((a, b))
    return pairs


def analyse(pairs_by_video, preds, mode, global_mode):
    def both(p, a, b):
        pa, pb = pred_of(p, a["unit_id"]), pred_of(p, b["unit_id"])
        return pa is not None and pb is not None and bucket(pa) == bucket(a["count"]) and bucket(pb) == bucket(b["count"])

    def prior(a, b):
        return (bucket(mode.get(a["action"], global_mode)) == bucket(a["count"]) and
                bucket(mode.get(b["action"], global_mode)) == bucket(b["count"]))

    flat = [(a, b) for rows in pairs_by_video.values() for a, b in rows]
    out = {"videos": len(pairs_by_video), "pairs": len(flat)}
    for name, p in (("zero-shot", preds["base"]), ("LoRA", preds["lora"]), ("readout", preds["head"])):
        hits = sum(1 for a, b in flat if both(p, a, b))
        out[name] = dict(both=hits, rate=round(hits / max(len(flat), 1), 4), ci=wilson(hits, len(flat)))
    hits = sum(1 for a, b in flat if prior(a, b))
    out["prior"] = dict(both=hits, rate=round(hits / max(len(flat), 1), 4), ci=wilson(hits, len(flat)))

    def delta(f, g):
        def one(videos):
            num = den = 0
            for v in videos:
                for a, b in pairs_by_video[v]:
                    den += 1
                    num += int(f(a, b)) - int(g(a, b))
            return num / den if den else 0.0
        videos = list(pairs_by_video)
        point = one(videos)
        rng = random.Random(0)
        draws = sorted(one([videos[rng.randrange(len(videos))] for _ in videos]) for _ in range(2000))
        below = sum(1 for d in draws if d <= 0)
        p = 2 * min(below, 2000 - below) / 2000
        return dict(delta=round(point, 4), ci=[round(draws[50], 4), round(draws[1949], 4)],
                    p=f"< {2/2000:.4f}" if p == 0 else f"{p:.4f}", videos=len(videos))

    out["tests"] = {
        "readout_minus_lora": delta(lambda a, b: both(preds["head"], a, b), lambda a, b: both(preds["lora"], a, b)),
        "readout_minus_prior": delta(lambda a, b: both(preds["head"], a, b), prior),
        "lora_minus_prior": delta(lambda a, b: both(preds["lora"], a, b), prior),
    }
    return out


def main():
    units = [json.loads(l) for l in UNITS.read_text(encoding="utf-8").splitlines()]
    train = [u for u in units if u["split"] == "train"]
    mode = {}
    for action in {u["action"] for u in train}:
        counter = Counter(bucket(u["count"]) for u in train if u["action"] == action)
        mode[action] = counter.most_common(1)[0][0]
    global_mode = Counter(bucket(u["count"]) for u in train).most_common(1)[0][0]

    report = {}
    pooled = defaultdict(list)
    for name, cfg in SPLITS.items():
        pairs = build_pairs(units if cfg["units"] == UNITS else
                            [json.loads(l) for l in VAL.read_text(encoding="utf-8").splitlines()],
                            cfg["split"])
        preds = {k: load(tag) for k, tag in cfg["preds"].items()}
        report[name] = analyse(pairs, preds, mode, global_mode)
        for video, rows in pairs.items():
            key = f"{name}:{video}"
            pooled[key] = rows
    # pooled over the two in-domain held-out splits (dev + test); VAL is a different sample
    preds = {"base": load("v3_test_base_32f"), "lora": load("v3_test_lora_32f"), "head": load("probe3_lora_test")}
    # ids do not collide across splits, so dev predictions can be merged in
    for tag, key in (("v3_dev_base_32f", "base"), ("v3_dev_lora_32f", "lora"), ("probe3_lora_dev", "head")):
        extra = load(tag)
        if extra:
            preds[key] = {**preds[key], **extra}
    pooled_pairs = defaultdict(list)
    for name in ("dev", "test"):
        cfg = SPLITS[name]
        pairs = build_pairs(units, cfg["split"])
        for video, rows in pairs.items():
            pooled_pairs[f"{name}:{video}"] = rows
    report["dev_plus_test"] = analyse(pooled_pairs, preds, mode, global_mode)

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
