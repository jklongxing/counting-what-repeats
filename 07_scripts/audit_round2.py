"""Second audit pass: remaining tables, same-video rates, calibration rows, and every
arithmetic identity asserted in the abstract and text.
"""
from __future__ import annotations

import json
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

B = Path(__file__).resolve().parents[1]
PRED = B / "02_predictions"
UNITS = [json.loads(l) for l in (B / "01_benchmark" / "evaluable_units.jsonl").read_text(encoding="utf-8").splitlines()]
VAL = [json.loads(l) for l in (B / "01_benchmark" / "val_units.jsonl").read_text(encoding="utf-8").splitlines()]
COUNTS = {u["unit_id"]: int(u["count"]) for u in UNITS + VAL}
SPLIT = {u["unit_id"]: u["split"] for u in UNITS + VAL}
PROBLEMS = []


def b4(c):
    return min(int(c), 4)


def pct(x):
    return float(Decimal(str(x * 100)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def load(tag):
    path = PRED / f"{tag}.jsonl"
    if not path.exists():
        return None
    return {json.loads(l)["unit_id"]: json.loads(l).get("pred") for l in path.read_text(encoding="utf-8").splitlines()}


def expect(label, got, want, tol=0.06):
    if got is None:
        PROBLEMS.append((label, "missing", want))
        return
    delta = abs(got - want)
    mark = "ok" if delta <= tol else "MISMATCH"
    print(f"{mark:8s} {label:44s} recomputed={got:6.1f} paper={want:6.1f}")
    if mark != "ok":
        PROBLEMS.append((label, got, want))


# ---------- Table 4.7: accuracy by true count ----------
print("== Table 4.7 (accuracy by true count, test split) ==")
claims = {
    "v3_test_base_32f": [52.0, 50.5, 20.5, 45.2],
    "v3_test_lora_32f": [91.9, 36.6, 46.2, 64.5],
    "probe3_lora_test": [88.2, 60.4, 48.7, 54.8],
}
for tag, wants in claims.items():
    preds = load(tag)
    for i, want in enumerate(wants, start=1):
        rows = [(p, COUNTS[u]) for u, p in preds.items() if u in COUNTS and SPLIT[u] == "test" and b4(COUNTS[u]) == i]
        got = sum(1 for p, c in rows if isinstance(p, int) and b4(p) == b4(c)) / len(rows) * 100 if rows else None
        expect(f"4.7 {tag} count={i}", pct(got / 100) if got is not None else None, want)

# ---------- Table 4.9: plane points ----------
print("== Table 4.9 (single vs repeated) ==")
plane = {
    "zero-shot Qwen2.5": ("v3_test_qwen25_base", 49.5, 22.2),
    "zero-shot Qwen3": ("v3_test_base_32f", 52.0, 42.7),
    "zero-shot InternVL3.5": ("v3_test_internvl", 66.7, 39.2),
    "class-weighted LoRA": ("v3_test_lora_bal", 85.0, 53.2),
    "MLP readout": ("probe3_lora_test", 88.2, 56.7),
    "LoRA": ("v3_test_lora_32f", 91.9, 43.9),
    "full fine-tuning": ("v3_test_fullft", 98.3, 32.2),
}
for name, (tag, want_s, want_r) in plane.items():
    preds = load(tag)
    rows = [(p, COUNTS[u]) for u, p in preds.items() if u in COUNTS and SPLIT[u] == "test"]
    once = [(p, c) for p, c in rows if c == 1]
    rep = [(p, c) for p, c in rows if c >= 2]
    s = sum(1 for p, _ in once if p == 1) / len(once) * 100
    r = sum(1 for p, c in rep if isinstance(p, int) and b4(p) == b4(c)) / len(rep) * 100
    expect(f"4.9 {name} single", pct(s / 100), want_s)
    expect(f"4.9 {name} repeated", pct(r / 100), want_r)

# ---------- Table 4.10: same-video both-correct rates ----------
print("== Table 4.10 (same-video pairs) ==")
by_video = defaultdict(list)
for u in UNITS:
    if u["split"] == "test":
        by_video[u["video_id"]].append(u)
pairs = []
for video, rows in by_video.items():
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a["action"] != b["action"] and a["count"] != b["count"]:
                pairs.append((a, b))
print(f"           pairs={len(pairs)} videos={len({v for v in by_video if any(a in by_video[v] for a, _ in pairs)})}")
for name, tag, want in (("zero-shot", "v3_test_base_32f", 25.4), ("LoRA", "v3_test_lora_32f", 28.7),
                        ("readout", "probe3_lora_test", 36.8)):
    preds = load(tag)
    hit = sum(1 for a, b in pairs
              if isinstance(preds.get(a["unit_id"]), int) and isinstance(preds.get(b["unit_id"]), int)
              and b4(preds[a["unit_id"]]) == b4(a["count"]) and b4(preds[b["unit_id"]]) == b4(b["count"]))
    expect(f"4.10 {name} both-correct", pct(hit / len(pairs)), want)

# ---------- arithmetic identities asserted in the abstract ----------
print("== arithmetic identities ==")
zs, prior_o, prior_s, prior_r = 49.2, 72.6, 95.8, 17.5
lora_o, lora_s, lora_r = 77.6, 91.9, 43.9
head_r, val_lora_r, val_head_r = 56.7, 30.1, 44.3
cal_r = 49.7
# unrounded operands, so the identity matches the paper's stated difference
identities = [
    ("prior - zero-shot = 23.4", round(72.62 - 49.22, 1), 23.4),
    ("LoRA - zero-shot overall = 28.4", round(77.64 - 49.22, 1), 28.4),
    ("LoRA - zero-shot repeated = 1.2", round(43.86 - 42.69, 1), 1.2),
    ("LoRA - prior repeated = 26.3", round(43.86 - 17.54, 1), 26.3),
    ("prior single - LoRA single = 3.9", round(95.81 - 91.87, 1), 3.9),
    ("readout - LoRA repeated = 12.9", round(56.73 - 43.86, 1), 12.9),
    ("VAL readout - VAL LoRA = 14.2", round(44.26 - 30.06, 1), 14.2),
    ("readout - matched calibration = 7.0", round(56.73 - 49.71, 1), 7.0),
]
for label, got, want in identities:
    mark = "ok" if abs(got - want) <= 0.05 else "MISMATCH"
    print(f"{mark:8s} {label:44s} computed={got:6.1f} paper={want:6.1f}")
    if mark != "ok":
        PROBLEMS.append((label, got, want))



# ---------- Section 4.11: same-question pairs and score-space head ----------
print("== Section 4.11 ==")
test_units = [u for u in UNITS if u["split"] == "test"]
by_action = defaultdict(lambda: {"once": [], "repeated": []})
for u in test_units:
    by_action[u["action"]]["once" if u["count"] == 1 else "repeated"].append(u)
sq = []
for action in sorted(by_action):
    once = sorted(by_action[action]["once"], key=lambda u: u["unit_id"])
    rep = sorted(by_action[action]["repeated"], key=lambda u: u["unit_id"])
    for a, b in zip(once, rep):
        if b4(a["count"]) != b4(b["count"]):
            sq.append((a, b))
print(f"           same-question pairs={len(sq)} actions={len({a['action'] for a, _ in sq})}")
for name, tag, want in (("zero-shot", "v3_test_base_32f", 21.5), ("LoRA", "v3_test_lora_32f", 30.0),
                        ("readout", "probe3_lora_test", 40.8)):
    preds = load(tag)
    hit = sum(1 for a, b in sq
              if isinstance(preds.get(a["unit_id"]), int) and isinstance(preds.get(b["unit_id"]), int)
              and b4(preds[a["unit_id"]]) == b4(a["count"]) and b4(preds[b["unit_id"]]) == b4(b["count"]))
    expect(f"4.11 {name} same-question", pct(hit / len(sq)), want)

for split, tag, wants in (("test", "scorehead_test", (77.1, 89.4, 48.0)),
                          ("val", "scorehead_val", (63.5, 91.0, 35.9))):
    preds = load(tag)
    rows = [(p, COUNTS[u]) for u, p in preds.items() if u in COUNTS and SPLIT[u] == split]
    once = [(p, c) for p, c in rows if c == 1]
    rep = [(p, c) for p, c in rows if c >= 2]
    overall = sum(1 for p, c in rows if isinstance(p, int) and b4(p) == b4(c)) / len(rows)
    single = sum(1 for p, c in once if p == 1) / len(once)
    repeated = sum(1 for p, c in rep if isinstance(p, int) and b4(p) == b4(c)) / len(rep)
    expect(f"4.11 score head {split} overall", pct(overall), wants[0])
    expect(f"4.11 score head {split} single", pct(single), wants[1])
    expect(f"4.11 score head {split} repeated", pct(repeated), wants[2])

print()
print("PROBLEMS:", len(PROBLEMS))
for p in PROBLEMS:
    print("  ", p)

