"""Two additions requested by review.

1. Same question, different videos. For each action we pair one single-occurrence video with
   one repeated video, one pair per unit, in a deterministic order fixed before looking at any
   prediction. A method that answers from the action name, or with a constant, must give the
   same answer to both questions and therefore cannot get both right.
2. Score-space readout. An MLP with the same configuration and training split as the
   hidden-state readout, but fed only the eight digit scores, to test whether the readout's
   advantage is just a more flexible classifier on the same information.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except ModuleNotFoundError:
    HAVE_SKLEARN = False

B = Path(__file__).resolve().parent / "DATA_BUNDLE_20261111"
PRED = B / "02_predictions"
UNITS = [json.loads(l) for l in (B / "01_benchmark" / "evaluable_units.jsonl").read_text(encoding="utf-8").splitlines()]
VAL = [json.loads(l) for l in (B / "01_benchmark" / "val_units.jsonl").read_text(encoding="utf-8").splitlines()]
OUT = B / "04_summaries" / "extra_controls.json"


def b4(c):
    return min(int(c), 4)


def load(tag):
    path = PRED / f"{tag}.jsonl"
    if not path.exists():
        return None
    return {json.loads(l)["unit_id"]: json.loads(l).get("pred") for l in path.read_text(encoding="utf-8").splitlines()}


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def same_question_pairs():
    test = [u for u in UNITS if u["split"] == "test"]
    by_action = defaultdict(lambda: {"once": [], "repeated": []})
    for u in test:
        key = "once" if u["count"] == 1 else "repeated"
        by_action[u["action"]][key].append(u)
    # deterministic pairing: sort by unit id, take the i-th from each side, one pair per unit
    pairs = []
    for action in sorted(by_action):
        once = sorted(by_action[action]["once"], key=lambda u: u["unit_id"])
        rep = sorted(by_action[action]["repeated"], key=lambda u: u["unit_id"])
        for a, b in zip(once, rep):
            if b4(a["count"]) != b4(b["count"]):
                pairs.append((a, b))
    return pairs


def main():
    pairs = same_question_pairs()
    preds = {"zero-shot": "v3_test_base_32f", "LoRA": "v3_test_lora_32f", "readout": "probe3_lora_test"}
    report = {"same_question": {"pairs": len(pairs),
                                "actions": len({a["action"] for a, _ in pairs}),
                                "videos": len({a["video_id"] for a, _ in pairs} | {b["video_id"] for _, b in pairs})}}
    for name, tag in preds.items():
        p = load(tag)
        hit = 0
        for a, b in pairs:
            pa, pb = p.get(a["unit_id"]), p.get(b["unit_id"])
            if isinstance(pa, int) and isinstance(pb, int) and b4(pa) == b4(a["count"]) and b4(pb) == b4(b["count"]):
                hit += 1
        report["same_question"][name] = dict(both=hit, rate=round(hit / max(len(pairs), 1), 4),
                                             ci=wilson(hit, len(pairs)))
    # baselines that see only the action
    prior = load("probe3_lora_test")  # placeholder, replaced below
    report["same_question"]["always1"] = dict(both=0, rate=0.0, ci=[0.0, 0.0])
    train = [u for u in UNITS if u["split"] == "train"]
    from collections import Counter
    mode = {}
    for action in {u["action"] for u in train}:
        c = Counter(b4(u["count"]) for u in train if u["action"] == action)
        mode[action] = c.most_common(1)[0][0]
    hit = sum(1 for a, b in pairs if b4(mode.get(a["action"], 1)) == b4(a["count"])
              and b4(mode.get(b["action"], 1)) == b4(b["count"]))
    report["same_question"]["per_action_prior"] = dict(both=hit, rate=round(hit / max(len(pairs), 1), 4),
                                                       ci=wilson(hit, len(pairs)))

    # ---- score-space readout ----
    if not HAVE_SKLEARN:
        report["score_head"] = {"skipped": "scikit-learn not available in this interpreter"}
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return
    scores = {}
    for shard in sorted((PRED.parent / "03_scores").glob("scores_lora_all_s*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            scores[row["unit_id"]] = row
    def pack(split):
        ids = [u["unit_id"] for u in UNITS + VAL if u["split"] == split and u["unit_id"] in scores]
        X = np.array([scores[i]["scores"] for i in ids], dtype=np.float64)
        y = np.array([b4(scores[i]["count"]) for i in ids])
        return ids, X, y
    ids_tr, X_tr, y_tr = pack("train")
    out = {"score_head": {"train": len(y_tr)}}
    for split, tag in (("test", "probe3_lora_test"), ("val", "probe3_lora_val")):
        ids, X, y = pack(split)
        results = []
        for seed in range(3):
            clf = make_pipeline(StandardScaler(),
                                MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=400,
                                              early_stopping=True, n_iter_no_change=15, random_state=seed))
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X)
            results.append(float((pred == y).mean()))
        out["score_head"][split] = dict(overall=round(float(np.mean(results)), 4),
                                        sd=round(float(np.std(results)), 4), n=len(y))
        # reference: hidden-state readout on the same ids
        ref = load(tag)
        ref_hit = sum(1 for i, yy in zip(ids, y) if isinstance(ref.get(i), int) and b4(ref[i]) == yy)
        out["score_head"][split]["hidden_readout"] = round(ref_hit / len(ids), 4)

    report.update(out)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
