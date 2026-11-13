"""Label-risk audit using the builder's own event filter.

An event is usable by build_benchmark_v3 exactly when its label is a non-empty string,
its timestamps are a valid pair, its parent_objects is a non-empty list, and every parent
has a valid id that resolves to a clock built from object_tracking. A unit whose action
has unusable events is labelled with fewer occurrences than the video contains.
"""
from __future__ import annotations

import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
UNITS = W / "p0_count_capacity" / "benchmark3_units.jsonl"
ZIP = W / "data" / "train_annotations.zip"
FRAMES = W / "data" / "frames"
OUT = W / "runs" / "label_risk.json"


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def valid_id(x):
    return type(x) is int and x >= 0


def main():
    with zipfile.ZipFile(ZIP) as archive:
        data = json.loads(archive.read("all_train.json").decode("utf-8-sig"))

    units = [json.loads(l) for l in UNITS.read_text(encoding="utf-8").splitlines()]
    units = [u for u in units if (FRAMES / u["video_id"]).exists()
             and len(list((FRAMES / u["video_id"]).glob("*.jpg"))) >= 32]

    risky = []
    for unit in units:
        record = data.get(unit["video_id"])
        if not record:
            continue
        n = record["metadata"]["num_frames"]
        clock = {}
        for track in record.get("object_tracking", []):
            nid = track.get("id")
            if not valid_id(nid) or nid in clock:
                continue
            frames, stamps = track.get("frame_ids"), track.get("timestamps")
            if not (isinstance(frames, list) and isinstance(stamps, list)
                    and len(frames) == len(stamps)):
                continue
            seen = {}
            for frame, stamp in zip(frames, stamps):
                if type(frame) is int and 0 <= frame < n and finite(stamp) and stamp >= 0:
                    seen.setdefault(frame, stamp)
            if seen:
                clock[nid] = seen

        total = usable = 0
        for event in record.get("action_localisation", []):
            if event.get("label") != unit["action"]:
                continue
            total += 1
            interval = event.get("timestamps")
            parents = event.get("parent_objects")
            ok = (isinstance(event.get("label"), str) and event["label"]
                  and isinstance(interval, list) and len(interval) == 2
                  and all(finite(x) for x in interval) and 0 <= interval[0] < interval[1]
                  and isinstance(parents, list) and parents
                  and all(valid_id(p) and p in clock for p in parents))
            usable += int(bool(ok))
        if total != usable:
            risky.append(dict(unit_id=unit["unit_id"], split=unit["split"], action=unit["action"],
                              events_in_video=total, usable_events=usable, label=unit["count"]))

    by_split = Counter(r["split"] for r in risky)
    summary = dict(units=len(units), risky=len(risky), by_split=dict(by_split), detail=risky)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "detail"}, ensure_ascii=False))
    for row in risky:
        print(row)


if __name__ == "__main__":
    main()
