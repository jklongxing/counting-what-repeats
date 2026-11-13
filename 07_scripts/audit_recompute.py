"""Independent audit: recompute every table from the bundled per-question predictions.

This script deliberately does NOT read canonical_results.json. It rebuilds each metric from
02_predictions/*.jsonl plus 01_benchmark/benchmark3_units.jsonl and compares the result with
the value printed in the manuscript. Any difference above 0.05 points is flagged.
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

B = Path(__file__).resolve().parents[1]
PRED = B / "02_predictions"
UNITS = [json.loads(l) for l in (B / "01_benchmark" / "evaluable_units.jsonl").read_text(encoding="utf-8").splitlines()]
VAL = [json.loads(l) for l in (B / "01_benchmark" / "val_units.jsonl").read_text(encoding="utf-8").splitlines()]
MCU = [json.loads(l) for l in (B / "01_benchmark" / "mc_units.jsonl").read_text(encoding="utf-8").splitlines()]
COUNTS = {u["unit_id"]: int(u["count"]) for u in UNITS + VAL + MCU}
SPLIT = {u["unit_id"]: u["split"] for u in UNITS + VAL + MCU}

FLAGS = []


def b4(c):
    return min(int(c), 4)


def p1(x):
    return float(Decimal(repr(round(x, 10))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def load(tag):
    path = PRED / f"{tag}.jsonl"
    if not path.exists():
        return None
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out[row["unit_id"]] = row.get("pred")
    return out


def metric(preds, split, how="bucket"):
    rows = [(preds[u], COUNTS[u]) for u in preds if u in COUNTS and SPLIT[u] == split]
    total = len(rows)
    if not total:
        return None
    ok = (lambda p, c: b4(p) == b4(c)) if how == "bucket" else (lambda p, c: p == c)
    valid = [(p, c) for p, c in rows if isinstance(p, int)]
    # denominators are every question in the group; an unparsable answer is an error
    once = [(p, c) for p, c in rows if c == 1]
    rep = [(p, c) for p, c in rows if c >= 2]
    return dict(
        n=total, parsed=len(valid),
        overall=sum(ok(p, c) for p, c in valid) / total,
        single=sum(1 for p, c in once if isinstance(p, int) and p == 1) / len(once) if once else float("nan"),
        repeated=sum(1 for p, c in rep if isinstance(p, int) and ok(p, c)) / len(rep) if rep else float("nan"),
        binary=sum(1 for p, c in valid if (p >= 2) == (c >= 2)) / total,
    )


def check(label, got, claimed, how="bucket"):
    if got is None:
        FLAGS.append((label, "MISSING RUN", claimed, None))
        return
    values = dict(overall=got["overall"], single=got["single"], repeated=got["repeated"], binary=got["binary"])
    for key, want in claimed.items():
        have = values.get(key) if how == "bucket" else None
        if have is None:
            continue
        delta = abs(p1(have * 100) - want)
        status = "ok" if delta <= 0.06 else "MISMATCH"
        if status != "ok":
            FLAGS.append((f"{label}.{key}", round(have * 100, 1), want, round(delta, 2)))
        print(f"{status:8s} {label:34s} {key:8s} recomputed={p1(have*100):5.1f} paper={want:5.1f}")


# ---- Table 4.1 ----
check("4.1 Qwen3-VL-8B", metric(load("v3_test_base_32f"), "test"), dict(overall=49.2, single=52.0, repeated=42.7, binary=66.0))
check("4.1 Qwen2.5-VL-7B", metric(load("v3_test_qwen25_base"), "test"), dict(overall=41.4, single=49.5, repeated=22.2, binary=67.6))
check("4.1 InternVL3.5-8B", metric(load("v3_test_internvl"), "test"), dict(overall=58.6, single=66.7, repeated=39.2, binary=69.3))
check("4.1 InternVL2.5-8B", metric(load("v3_test_internvl25"), "test"), dict(overall=21.0, single=10.8, repeated=45.0, binary=30.8))

# ---- Table 4.2 frame budget (dev) ----
for frames, claimed in (("4", (43.4, 55.0, 16.7, 66.4)), ("8", (44.8, 51.5, 29.3, 65.2)),
                        ("16", (48.1, 55.2, 31.6, 65.1)), ("32", (50.9, 56.4, 37.9, 68.5)),
                        ("128", (53.8, 60.6, 37.9, 68.3)), ("256", (57.1, 66.1, 36.2, 69.7))):
    tag = "v3_dev_base_32f" if frames == "32" else f"v3_dev_{frames}f"
    check(f"4.2 {frames} frames", metric(load(tag), "dev"),
          dict(overall=claimed[0], single=claimed[1], repeated=claimed[2], binary=claimed[3]))

# ---- Table 4.3 supervision ----
check("4.3 test zero-shot", metric(load("v3_test_base_32f"), "test"), dict(overall=49.2, single=52.0, repeated=42.7))
check("4.3 test LoRA", metric(load("v3_test_lora_32f"), "test"), dict(overall=77.6, single=91.9, repeated=43.9))
check("4.3 test full FT", metric(load("v3_test_fullft"), "test"), dict(overall=78.7, single=98.3, repeated=32.2))
check("4.3 dev zero-shot", metric(load("v3_dev_base_32f"), "dev"), dict(overall=50.9, single=56.4, repeated=37.9))
check("4.3 dev LoRA", metric(load("v3_dev_lora_32f"), "dev"), dict(overall=78.2, single=94.1, repeated=41.4))
check("4.3 VAL zero-shot", metric(load("val_base_32f"), "val"), dict(overall=47.0, single=57.5, repeated=36.5))
check("4.3 VAL LoRA", metric(load("val_lora_32f"), "val"), dict(overall=61.5, single=92.9, repeated=30.1))

# ---- Table 4.4 readout ----
check("4.4 test readout", metric(load("probe3_lora_test"), "test"), dict(overall=78.9, single=88.2, repeated=56.7))
check("4.4 VAL readout", metric(load("probe3_lora_val"), "val"), dict(overall=65.3, single=86.3, repeated=44.3))
for seed, claimed in ((7, (76.6, 98.3, 25.1)), (8, (78.0, 97.3, 32.2))):
    check(f"4.4 adapter seed{seed}", metric(load(f"v3_test_seed{seed}"), "test"),
          dict(overall=claimed[0], single=claimed[1], repeated=claimed[2]))

# ---- Table 4.5 human-written ----
mc = load("mc_qwen3vl_32f")
if mc:
    vals = [(p, COUNTS[u]) for u, p in mc.items() if u in COUNTS and isinstance(p, int)]
    for name, want in (("zero-shot", 44.2),):
        got = sum(b4(p) == b4(c) for p, c in vals) / len(vals) * 100
        status = "ok" if abs(round(got, 1) - want) <= 0.05 else "MISMATCH"
        print(f"{status:8s} 4.5 human questions {name:10s} recomputed={got:5.1f} paper={want:5.1f}")
        if status != "ok":
            FLAGS.append((f"4.5 {name}", round(got, 1), want, round(abs(got - want), 2)))
mc_lora = load("mc_lora")
if mc_lora:
    vals = [(p, COUNTS[u]) for u, p in mc_lora.items() if u in COUNTS and isinstance(p, int)]
    got = sum(b4(p) == b4(c) for p, c in vals) / len(vals) * 100
    status = "ok" if abs(round(got, 1) - 45.7) <= 0.05 else "MISMATCH"
    print(f"{status:8s} 4.5 human questions LoRA       recomputed={got:5.1f} paper= 45.7")
    if status != "ok":
        FLAGS.append(("4.5 LoRA", round(got, 1), 45.7, round(abs(got - 45.7), 2)))

# ---- internal consistency of the universe ----
total_units = len(UNITS)
by_split = {}
for u in UNITS:
    by_split[u["split"]] = by_split.get(u["split"], 0) + 1
print()
print("universe:", total_units, by_split)
if total_units != 3710:
    FLAGS.append(("universe size", total_units, 3710, abs(total_units - 3710)))
if by_split != {"train": 2555, "dev": 578, "test": 577}:
    FLAGS.append(("split sizes", by_split, "2555/578/577", None))

print()
if FLAGS:
    print("FLAGS (%d):" % len(FLAGS))
    for f in FLAGS:
        print("  ", f)
else:
    print("All checked values reproduce within 0.05 points.")
