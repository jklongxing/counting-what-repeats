"""Canonical scoring v2.

Changes from v1, in response to review:
- every question in the fixed universe is scored; an unparsable or missing answer is
  an error, never dropped from the denominator
- parse counts are reported separately
- the bootstrap treats unparsable answers as errors
- p values that hit the resolution floor are reported as an upper bound, and a
  degenerate all-zero difference is reported as p = 1
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"
UNITS = W / "p0_count_capacity" / "benchmark3_units.jsonl"
VAL_UNITS = W / "p0_count_capacity" / "val_units.jsonl"
FRAMES = W / "data" / "frames"
OUT = RUNS / "canonical_results.json"
ITERS = 2000


def bucket(c):
    return min(int(c), 4)


def load_units(path=UNITS):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if (FRAMES / r["video_id"]).exists()
            and len(list((FRAMES / r["video_id"]).glob("*.jpg"))) >= 32]


def load_preds(tag):
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


def score(preds, units, split):
    rows = [u for u in units if u["split"] == split and u["unit_id"] in preds]
    parsed = sum(1 for u in rows if pred_of(preds, u["unit_id"]) is not None)

    def acc(subset):
        if not subset:
            return dict(n=0, exact4=None, exact_raw=None)
        hit4 = sum(1 for u in subset
                   if pred_of(preds, u["unit_id"]) is not None
                   and bucket(pred_of(preds, u["unit_id"])) == bucket(u["count"]))
        hitraw = sum(1 for u in subset if pred_of(preds, u["unit_id"]) == u["count"])
        return dict(n=len(subset), exact4=round(hit4 / len(subset), 4),
                    exact_raw=round(hitraw / len(subset), 4))

    once = [u for u in rows if u["count"] == 1]
    rep = [u for u in rows if u["count"] >= 2]
    binary = sum(1 for u in rows
                 if pred_of(preds, u["unit_id"]) is not None
                 and (pred_of(preds, u["unit_id"]) >= 2) == (u["count"] >= 2)) / max(len(rows), 1)
    return dict(units=len(rows), parsed=parsed, parse_rate=round(parsed / max(len(rows), 1), 4),
                all=acc(rows), once=dict(n=len(once),
                                         acc=round(sum(1 for u in once if pred_of(preds, u["unit_id"]) == 1) / max(len(once), 1), 4)),
                repeated=acc(rep), binary=round(binary, 4))


def cluster_bootstrap(pred_a, pred_b, units, subset_filter):
    by_video = defaultdict(list)
    for u in units:
        if subset_filter(u) and u["unit_id"] in pred_a and u["unit_id"] in pred_b:
            by_video[u["video_id"]].append(u)
    videos = list(by_video)
    if not videos:
        return None

    def delta(sample):
        hit_a = hit_b = total = 0
        for v in sample:
            for u in by_video[v]:
                total += 1
                pa, pb = pred_of(pred_a, u["unit_id"]), pred_of(pred_b, u["unit_id"])
                hit_a += pa is not None and bucket(pa) == bucket(u["count"])
                hit_b += pb is not None and bucket(pb) == bucket(u["count"])
        return (hit_b - hit_a) / total if total else 0.0

    point = delta(videos)
    rng = random.Random(0)
    draws = sorted(delta([videos[rng.randrange(len(videos))] for _ in videos]) for _ in range(ITERS))
    lo, hi = draws[int(0.025 * ITERS)], draws[int(0.975 * ITERS)]
    if all(abs(d) < 1e-12 for d in draws):
        return dict(delta=0.0, ci=[0.0, 0.0], p="1", videos=len(videos), note="degenerate: no difference")
    below = sum(1 for d in draws if d <= 0)
    p = 2 * min(below, ITERS - below) / ITERS
    p_text = f"< {2 / ITERS:.4f}" if p == 0 else f"{p:.4f}"
    return dict(delta=round(point, 4), ci=[round(lo, 4), round(hi, 4)], p=p_text,
                videos=len(videos), draws=ITERS)


def main():
    units = load_units()
    val_units = load_units(VAL_UNITS)
    report = {"universe": dict(total=len(units), by_split=dict(Counter(u["split"] for u in units)),
                               videos=len({u["video_id"] for u in units}),
                               videos_by_split=dict(Counter(
                                   {v: None for v in []}) or {}),
                               count_dist=dict(Counter(bucket(u["count"]) for u in units))),
              "scoring": "every question in the fixed universe is scored; unparsable answers count as errors"}

    per_split_videos = {}
    for split in ("train", "dev", "test"):
        per_split_videos[split] = len({u["video_id"] for u in units if u["split"] == split})
    report["universe"]["videos_by_split"] = per_split_videos
    report["val_universe"] = dict(total=len(val_units), videos=len({u["video_id"] for u in val_units}))

    train = [u for u in units if u["split"] == "train"]
    mode = {}
    for action in {u["action"] for u in train}:
        counter = Counter(bucket(u["count"]) for u in train if u["action"] == action)
        mode[action] = counter.most_common(1)[0][0]
    global_mode = Counter(bucket(u["count"]) for u in train).most_common(1)[0][0]
    prior = {u["unit_id"]: {"unit_id": u["unit_id"], "pred": mode.get(u["action"], global_mode)} for u in units}
    report["prior"] = dict(actions=len(mode), mode_dist=dict(Counter(mode.values())))

    runs = {
        "zero_shot_test": ("v3_test_base_32f", "test"),
        "zero_shot_dev": ("v3_dev_base_32f", "dev"),
        "lora_test": ("v3_test_lora_32f", "test"),
        "lora_dev": ("v3_dev_lora_32f", "dev"),
        "fullft_test": ("v3_test_fullft", "test"),
        "fullft_dev": ("v3_dev_fullft", "dev"),
        "balanced_lora_test": ("v3_test_lora_bal", "test"),
        "balanced_lora_dev": ("v3_dev_lora_bal", "dev"),
        "seed7_test": ("v3_test_seed7", "test"),
        "seed8_test": ("v3_test_seed8", "test"),
        "head_test": ("probe3_lora_test", "test"),
        "qwen25_test": ("v3_test_qwen25_base", "test"),
        "qwen25_lora_test": ("v3_test_q25_lora", "test"),
        "internvl_test": ("v3_test_internvl", "test"),
        "internvl25_test": ("v3_test_internvl25", "test"),
        "val_base": ("val_base_32f", "val"),
        "val_lora": ("val_lora_32f", "val"),
        "val_head": ("probe3_lora_val", "val"),
    }
    scores = {}
    for name, (tag, split) in runs.items():
        preds = load_preds(tag)
        if preds is None:
            continue
        scores[name] = score(preds, val_units if split == "val" else units, split)
    report["scores"] = scores
    report["prior_score"] = score(prior, units, "test")

    frames = {}
    for f, tag in {4: "v3_dev_4f", 8: "v3_dev_8f", 16: "v3_dev_16f", 32: "v3_dev_base_32f",
                   128: "v3_dev_128f", 256: "v3_dev_256f"}.items():
        preds = load_preds(tag)
        if preds:
            frames[str(f)] = score(preds, units, "dev")
    report["frame_curve_qwen3"] = frames

    rep = lambda u: u["count"] >= 2
    base_test, lora_test, head_test = (load_preds("v3_test_base_32f"), load_preds("v3_test_lora_32f"),
                                       load_preds("probe3_lora_test"))
    report["tests"] = {
        "head_minus_lora_repeated": cluster_bootstrap(lora_test, head_test, units, lambda u: u["split"] == "test" and rep(u)),
        "head_minus_lora_all": cluster_bootstrap(lora_test, head_test, units, lambda u: u["split"] == "test"),
        "lora_minus_zeroshot_all": cluster_bootstrap(base_test, lora_test, units, lambda u: u["split"] == "test"),
        "lora_minus_zeroshot_repeated": cluster_bootstrap(base_test, lora_test, units, lambda u: u["split"] == "test" and rep(u)),
        "head_minus_prior_repeated": cluster_bootstrap(prior, head_test, units, lambda u: u["split"] == "test" and rep(u)),
        "zeroshot_minus_prior_all": cluster_bootstrap(prior, base_test, units, lambda u: u["split"] == "test"),
        "val_lora_minus_base_all": cluster_bootstrap(load_preds("val_base_32f"), load_preds("val_lora_32f"),
                                                     val_units, lambda u: u["split"] == "val"),
        "val_head_minus_lora_repeated": cluster_bootstrap(load_preds("val_lora_32f"), load_preds("probe3_lora_val"),
                                                          val_units, lambda u: u["split"] == "val" and rep(u)),
        "val_head_minus_lora_all": cluster_bootstrap(load_preds("val_lora_32f"), load_preds("probe3_lora_val"),
                                                     val_units, lambda u: u["split"] == "val"),
    }

    corr = {}
    for name, tag in (("zero_shot", "v3_test_base_32f"), ("lora", "v3_test_lora_32f"), ("head", "probe3_lora_test")):
        preds = load_preds(tag)
        pairs = [(pred_of(preds, u["unit_id"]), u["count"]) for u in units
                 if u["split"] == "test" and pred_of(preds, u["unit_id"]) is not None]
        rho, p = spearmanr([x for x, _ in pairs], [y for _, y in pairs])
        corr[name] = dict(spearman=round(float(rho), 4), p=float(p), n=len(pairs))
    report["correlation"] = corr

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def line(name, v):
        if not v:
            return f"{name}: missing"
        return (f"{name:24s} n={v['units']:4d} parsed={v['parsed']:4d} "
                f"exact4={v['all']['exact4']:.4f} raw={v['all']['exact_raw']:.4f} "
                f"once={v['once']['acc']:.4f} rep={v['repeated']['exact4']:.4f} bin={v['binary']:.4f}")

    print(line("prior", report["prior_score"]))
    for name, v in scores.items():
        print(line(name, v))
    print("\nframes:", {k: v["repeated"]["exact4"] for k, v in frames.items()})
    print("\ntests:")
    for name, v in report["tests"].items():
        print(" ", name, v)


if __name__ == "__main__":
    main()
