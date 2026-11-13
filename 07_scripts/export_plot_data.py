"""Export everything the local figure pipeline needs into one JSON file."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"
FRAMES = W / "data" / "frames"
UNITS = W / "p0_count_capacity" / "benchmark3_units.jsonl"


def bucket(c):
    return min(int(c), 4)


def load(tag):
    path = RUNS / f"{tag}.jsonl"
    if not path.exists():
        return None
    return {r["unit_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())}


def repeated_acc(preds, units):
    if not preds:
        return None
    hits = n = 0
    for u in units:
        if u["count"] < 2 or u["unit_id"] not in preds:
            continue
        p = preds[u["unit_id"]].get("pred")
        if not isinstance(p, int):
            continue
        n += 1
        hits += bucket(p) == bucket(u["count"])
    return round(100 * hits / n, 2) if n else None


def main():
    units = [json.loads(l) for l in UNITS.read_text(encoding="utf-8").splitlines()]
    units = [u for u in units if (FRAMES / u["video_id"]).exists()
             and len(list((FRAMES / u["video_id"]).glob("*.jpg"))) >= 32]
    dev = [u for u in units if u["split"] == "dev"]

    out = {}
    out["count_dist"] = dict(Counter(bucket(u["count"]) for u in units))
    out["splits"] = dict(Counter(u["split"] for u in units))
    out["total"] = len(units)

    frame_runs = {
        "Qwen3-VL-8B": {4: "v3_dev_4f", 8: "v3_dev_8f", 16: "v3_dev_16f", 32: "v3_dev_base_32f",
                        128: "v3_dev_128f", 256: "v3_dev_256f"},
        "Qwen2.5-VL-7B": {4: "v3_dev_q25_4f", 8: "v3_dev_q25_8f", 16: "v3_dev_q25_16f",
                          32: "v3_dev_qwen25_base", 128: "v3_dev_q25_128f"},
        "InternVL3.5-8B": {4: "v3_dev_ivl_4f", 8: "v3_dev_ivl_8f", 16: "v3_dev_ivl_16f",
                           32: "v3_dev_internvl"},
    }
    out["frame_curve"] = {}
    for model, points in frame_runs.items():
        out["frame_curve"][model] = {}
        for frames, tag in points.items():
            value = repeated_acc(load(tag), dev)
            if value is not None:
                out["frame_curve"][model][frames] = value

    canon = json.loads((RUNS / "canonical_results.json").read_text(encoding="utf-8"))
    out["canonical"] = canon
    for extra in ("multiseed_probe.json", "error_structure.json", "review_analyses.json", "head_ablation.json"):
        path = RUNS / extra
        if path.exists():
            out[extra.replace(".json", "")] = json.loads(path.read_text(encoding="utf-8"))

    target = RUNS / "plot_data.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", target, target.stat().st_size, "bytes")
    print("frame_curve", json.dumps(out["frame_curve"], ensure_ascii=False))


if __name__ == "__main__":
    main()
