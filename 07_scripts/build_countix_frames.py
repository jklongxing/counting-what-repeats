"""Build a Countix external test set: clips with 2-4 repetitions, 32-frame grids, units file.

Countix contains only repeated actions (minimum count 2), so this set tests repeated counting
on a different source with the dataset's own question phrasing; it has no single-occurrence
questions and therefore cannot reproduce the single/repeated decomposition.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import av
import numpy as np
from PIL import Image

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
ROOT = W / "data" / "countix_qa"
FRAMES = W / "data" / "frames_countix"
CLIPS = ROOT / "clips"
OUT = W / "p0_count_capacity" / "countix_units.jsonl"


def fixed_grid(n, k=32):
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def main():
    parsed = json.loads((ROOT / "countix_parsed.json").read_text(encoding="utf-8"))
    rows = [r for r in parsed["val"] if r["count"] <= 4]
    print("candidates:", len(rows))
    CLIPS.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)

    zpath = ROOT / "cut_coutix_val.zip"
    units = []
    with zipfile.ZipFile(zpath) as archive:
        names = {Path(n).name: n for n in archive.namelist() if n.endswith(".mp4")}
        for row in rows:
            name = Path(row["video"]).name
            member = names.get(name)
            if member is None:
                continue
            vid = name[:-4]
            frame_dir = FRAMES / vid
            if not frame_dir.exists() or len(list(frame_dir.glob("*.jpg"))) < 32:
                clip = CLIPS / name
                if not clip.exists():
                    clip.write_bytes(archive.read(member))
                frames = []
                with av.open(str(clip)) as container:
                    for frame in container.decode(container.streams.video[0]):
                        frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) < 8:
                    continue
                grid = fixed_grid(len(frames))
                frame_dir.mkdir(parents=True, exist_ok=True)
                for slot, index in enumerate(grid):
                    Image.fromarray(frames[index]).save(frame_dir / f"{slot:02d}.jpg", quality=95)
            units.append(dict(
                unit_id=f"{vid}|countix", video_id=vid, action="repeated action",
                split="countix", count=int(row["count"]), tier="repeated", balanced=False,
                num_frames=0,
                question="How often is the repeated action performed in the video? Answer with a single integer.",
                answer=str(row["count"])))

    OUT.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in units), encoding="utf-8")
    from collections import Counter
    print("units:", len(units), "frames dirs:", len(list(FRAMES.glob('*'))))
    print("counts:", Counter(u["count"] for u in units).most_common())


if __name__ == "__main__":
    main()
