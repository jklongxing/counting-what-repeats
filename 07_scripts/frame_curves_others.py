"""Four-way frame curves for Qwen2.5 and InternVL on the dev split."""
import json
import os

W = "/home/jiquan/video-state-evidence/repetition_readout_v1/"
units = [json.loads(l) for l in open(W + "p0_count_capacity/benchmark3_units.jsonl")]
units = [u for u in units if os.path.exists(W + "data/frames/" + u["video_id"])
         and len(os.listdir(W + "data/frames/" + u["video_id"])) >= 32]
dev = [u for u in units if u["split"] == "dev"]


def b(c):
    return min(int(c), 4)


def score(tag):
    path = W + "runs/" + tag + ".jsonl"
    if not os.path.exists(path):
        return None
    d = {r["unit_id"]: r for r in (json.loads(l) for l in open(path))}
    rows = [u for u in dev if u["unit_id"] in d]
    hit = sum(1 for u in rows if isinstance(d[u["unit_id"]].get("pred"), int)
              and b(d[u["unit_id"]]["pred"]) == b(u["count"]))
    rep = [u for u in rows if u["count"] >= 2]
    rh = sum(1 for u in rep if isinstance(d[u["unit_id"]].get("pred"), int)
             and b(d[u["unit_id"]]["pred"]) == b(u["count"]))
    return dict(n=len(rows), overall=round(hit / len(rows), 4), repeated=round(rh / len(rep), 4))


sets = {
    "qwen25": {4: "v3_dev_q25_4f", 8: "v3_dev_q25_8f", 16: "v3_dev_q25_16f",
               32: "v3_dev_qwen25_base", 128: "v3_dev_q25_128f"},
    "internvl": {4: "v3_dev_ivl_4f", 8: "v3_dev_ivl_8f", 16: "v3_dev_ivl_16f",
                 32: "v3_dev_internvl"},
}
out = {}
for name, tags in sets.items():
    out[name] = {}
    for frames, tag in tags.items():
        got = score(tag)
        out[name][frames] = got
        print(name, frames, got)
json.dump(out, open(W + "runs/frame_curves_other_models.json", "w"), indent=1)
