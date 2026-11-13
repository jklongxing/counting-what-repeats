"""Same-video groups: how many questions can be formed, and the all-correct rate.

The pair experiment used two questions per video. This script reports the distribution of
group sizes and, where a video supports three or four distinct actions with distinct
counts, the fraction of videos whose answers are all correct.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"
UNITS = W / "p0_count_capacity" / "benchmark3_units.jsonl"
FRAMES = W / "data" / "frames"
OUT = RUNS / "same_video_groups.json"


def bucket(c):
    return min(int(c), 4)


def load_units():
    rows = [json.loads(l) for l in UNITS.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if (FRAMES / r["video_id"]).exists()
            and len(list((FRAMES / r["video_id"]).glob("*.jpg"))) >= 32]


def load(tag):
    path = RUNS / f"{tag}.jsonl"
    return {r["unit_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())} if path.exists() else None


def pred_of(preds, uid):
    row = preds.get(uid)
    if not row:
        return None
    v = row.get("pred")
    return v if isinstance(v, int) else None


def main():
    units = load_units()
    train = [u for u in units if u["split"] == "train"]
    mode = {}
    for action in {u["action"] for u in train}:
        c = Counter(bucket(u["count"]) for u in train if u["action"] == action)
        mode[action] = c.most_common(1)[0][0]

    test = [u for u in units if u["split"] == "test"]
    by_video = defaultdict(list)
    for u in test:
        by_video[u["video_id"]].append(u)

    # keep only groups of distinct actions with distinct counts
    groups = {}
    for video, rows in by_video.items():
        seen_actions, seen_counts, keep = set(), set(), []
        for u in sorted(rows, key=lambda r: r["count"]):
            if u["action"] in seen_actions or u["count"] in seen_counts:
                continue
            seen_actions.add(u["action"])
            seen_counts.add(u["count"])
            keep.append(u)
        if len(keep) >= 2:
            groups[video] = keep

    size_dist = Counter(len(g) for g in groups.values())
    lora, head, base = load("v3_test_lora_32f"), load("probe3_lora_test"), load("v3_test_base_32f")

    def all_correct(preds, rows):
        return all(pred_of(preds, u["unit_id"]) is not None
                   and bucket(pred_of(preds, u["unit_id"])) == bucket(u["count"]) for u in rows)

    out = {"size_distribution": dict(size_dist), "videos": len(groups), "by_k": {}}
    for k in sorted(size_dist):
        subset = [g for g in groups.values() if len(g) == k]
        entry = {"videos": len(subset), "questions": sum(len(g) for g in subset)}
        for name, preds in (("zero-shot", base), ("LoRA", lora), ("readout", head)):
            entry[name] = round(sum(all_correct(preds, g) for g in subset) / len(subset), 4)
        entry["prior"] = round(sum(all(int(mode.get(u["action"], 1) == u["count"]) for u in g)
                                   for g in subset) / len(subset), 4)
        entry["always1"] = round(sum(all(u["count"] == 1 for u in g) for g in subset) / len(subset), 4)
        out["by_k"][str(k)] = entry

    # pooled: restrict to groups of exactly 4 where available, else 3
    for target in (4, 3):
        subset = [g for g in groups.values() if len(g) == target]
        if subset:
            out[f"all{target}"] = {
                "videos": len(subset),
                "zero-shot": round(sum(all_correct(base, g) for g in subset) / len(subset), 4),
                "LoRA": round(sum(all_correct(lora, g) for g in subset) / len(subset), 4),
                "readout": round(sum(all_correct(head, g) for g in subset) / len(subset), 4),
                "prior": round(sum(all(int(mode.get(u["action"], 1) == u["count"]) for u in g)
                                   for g in subset) / len(subset), 4),
            }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
