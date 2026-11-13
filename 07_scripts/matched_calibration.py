"""Matched-operating-point calibration and the prior-bucketing fix.

1. Calibration: from the saved 8-way digit scores, add one common bias beta to every
   candidate answer >= 2 (a repetition bias), sweep beta on dev, pick the beta whose dev
   single-occurrence accuracy is closest to the readout's dev single-occurrence accuracy,
   freeze it, then report test and VAL with a video-clustered paired test against the readout.
2. Group table: score the per-action prior with the same 4-way bucketing as everything else.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"
UNITS = W / "p0_count_capacity" / "benchmark3_units.jsonl"
FRAMES = W / "data" / "frames"
OUT = RUNS / "matched_calibration_and_groups.json"


def bucket(c):
    return min(int(c), 4)


def load_scores():
    rows = {}
    for path in sorted(RUNS.glob("scores_lora_all_s*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[row["unit_id"]] = row
    return rows


def load_preds(tag):
    path = RUNS / f"{tag}.jsonl"
    return {r["unit_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())} if path.exists() else None


def evaluate(rows, beta, mask_repeat):
    pred, truth = [], []
    for row in rows:
        s = np.array(row["scores"], dtype=np.float64)
        if mask_repeat:
            s = s + beta * np.arange(1, 9).__ge__(2)
        pred.append(int(np.argmax(s)) + 1)
        truth.append(int(row["count"]))
    pred = np.array(pred)
    truth = np.array(truth)
    hit = np.minimum(pred, 4) == np.minimum(truth, 4)
    once = truth == 1
    rep = truth >= 2
    return dict(overall=float(hit.mean()),
                single=float(hit[once].mean()) if once.any() else None,
                repeated=float(hit[rep].mean()) if rep.any() else None), pred, truth


def main():
    scores = load_scores()
    by_split = defaultdict(list)
    for row in scores.values():
        by_split[row["split"]].append(row)

    readout_dev = json.loads((RUNS / "probe3_lora_dev.jsonl").read_text(encoding="utf-8").splitlines()[0]) if False else None
    dev_rows = by_split["dev"]
    # readout dev single accuracy, computed from its own predictions
    rp = load_preds("probe3_lora_dev")
    dev_truth = np.array([int(r["count"]) for r in dev_rows])
    readout_single = None
    if rp:
        hits = [int(bucket(rp[r["unit_id"]]["pred"]) == 1) for r in dev_rows
                if r["unit_id"] in rp and isinstance(rp[r["unit_id"]].get("pred"), int) and int(r["count"]) == 1]
        readout_single = sum(hits) / len(hits) if hits else None

    grid = [round(x * 0.25, 2) for x in range(-8, 25)]
    dev_curve = {}
    for beta in grid:
        stats, _, _ = evaluate(dev_rows, beta, True)
        dev_curve[beta] = stats
    best_beta = min(grid, key=lambda b: abs(dev_curve[b]["single"] - readout_single))

    report = dict(readout_dev_single=round(readout_single, 4), chosen_beta=best_beta,
                  dev_matched=dev_curve[best_beta],
                  dev_matched_error=round(abs(dev_curve[best_beta]["single"] - readout_single), 4),
                  dev_curve={str(k): v for k, v in dev_curve.items()})
    for split in ("test", "val"):
        rows = by_split[split]
        if not rows:
            continue
        stats, pred, truth = evaluate(rows, best_beta, True)
        base, _, _ = evaluate(rows, 0.0, False)
        report[split] = dict(matched=stats, no_bias=base)

        # clustered paired test against the readout
        tag = "probe3_lora_test" if split == "test" else "probe3_lora_val"
        rpreds = load_preds(tag)
        by_video = defaultdict(list)
        for row, p in zip(rows, pred):
            if row["unit_id"] in rpreds and isinstance(rpreds[row["unit_id"]].get("pred"), int):
                by_video[row["video_id"]].append((int(row["count"]), int(p), int(rpreds[row["unit_id"]]["pred"])))

        def delta(sample):
            num = den = 0
            for v in sample:
                for count, p_cal, p_ro in by_video[v]:
                    if count < 2:
                        continue
                    den += 1
                    num += int(bucket(p_ro) == bucket(count)) - int(bucket(p_cal) == bucket(count))
            return num / den if den else 0.0

        videos = list(by_video)
        if videos:
            point = delta(videos)
            rng = random.Random(0)
            draws = sorted(delta([videos[rng.randrange(len(videos))] for _ in videos]) for _ in range(2000))
            below = sum(1 for d in draws if d <= 0)
            p = 2 * min(below, 2000 - below) / 2000
            report[split]["readout_minus_calibrated_repeated"] = dict(
                delta=round(point, 4), ci=[round(draws[50], 4), round(draws[1949], 4)],
                p=f"< {2/2000:.4f}" if p == 0 else f"{p:.4f}", videos=len(videos))

    # ---- group table with bucketed prior ----
    units = [json.loads(l) for l in UNITS.read_text(encoding="utf-8").splitlines()]
    units = [u for u in units if (FRAMES / u["video_id"]).exists()
             and len(list((FRAMES / u["video_id"]).glob("*.jpg"))) >= 32]
    train = [u for u in units if u["split"] == "train"]
    mode = {}
    for action in {u["action"] for u in train}:
        counter = Counter(bucket(u["count"]) for u in train if u["action"] == action)
        mode[action] = counter.most_common(1)[0][0]
    test = [u for u in units if u["split"] == "test"]
    by_video = defaultdict(list)
    for u in test:
        by_video[u["video_id"]].append(u)
    groups = {}
    for video, rows in by_video.items():
        seen_a, seen_c, keep = set(), set(), []
        for u in sorted(rows, key=lambda r: r["count"]):
            if u["action"] in seen_a or u["count"] in seen_c:
                continue
            seen_a.add(u["action"])
            seen_c.add(u["count"])
            keep.append(u)
        if len(keep) >= 2:
            groups[video] = keep

    preds = {"zero-shot": load_preds("v3_test_base_32f"), "LoRA": load_preds("v3_test_lora_32f"),
             "readout": load_preds("probe3_lora_test")}
    table = {}
    for k in sorted({len(g) for g in groups.values()}):
        subset = [g for g in groups.values() if len(g) == k]
        row = {"videos": len(subset)}
        for name, p in preds.items():
            row[name] = round(sum(all(isinstance(p[u["unit_id"]].get("pred"), int)
                                     and bucket(p[u["unit_id"]]["pred"]) == bucket(u["count"]) for u in g)
                                 for g in subset) / len(subset), 4)
        row["prior_bucketed"] = round(sum(all(bucket(mode.get(u["action"], 1)) == bucket(u["count"]) for u in g)
                                          for g in subset) / len(subset), 4)
        row["always1"] = round(sum(all(bucket(1) == bucket(u["count"]) for u in g) for g in subset) / len(subset), 4)
        table[str(k)] = row
    report["groups_test"] = table

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "dev_curve"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
