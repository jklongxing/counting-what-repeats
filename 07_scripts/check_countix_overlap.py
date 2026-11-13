"""Check the actual train/val video overlap for the Countix refit."""
import json
from collections import Counter
from pathlib import Path

import torch

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"


def ids(name):
    blob = torch.load(RUNS / name, map_location="cpu", weights_only=False)
    rows = blob["units"]
    return [(r["unit_id"], r["video_id"], int(r["count"])) for r in rows]


def main():
    tr = ids("h3_countix_train_Qwen3-VL-8B-Instruct_32f.pt")
    va = ids("h3_lora_countix_Qwen3-VL-8B-Instruct_32f.pt")
    tr_v = {v for _, v, _ in tr}
    va_v = {v for _, v, _ in va}
    inter = tr_v & va_v
    print("train units", len(tr), "videos", len(tr_v), Counter(c for _, _, c in tr).most_common())
    print("val units", len(va), "videos", len(va_v))
    print("video overlap:", len(inter), sorted(inter))
    report = dict(train_units=len(tr), train_videos=len(tr_v), val_units=len(va), val_videos=len(va_v),
                  overlap_videos=sorted(inter), overlap_units=[u for u, v, _ in tr if v in inter])
    (RUNS / "countix_overlap.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                               encoding="utf-8")
    (RUNS / "countix_train_ids.json").write_text(
        json.dumps([dict(unit_id=u, video_id=v, count=c) for u, v, c in tr]), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
