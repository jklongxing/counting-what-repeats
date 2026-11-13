"""Prepare Countix training clips (count<=4): download the train zip, extract 32-frame grids."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import av
from PIL import Image
from huggingface_hub import hf_hub_download

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
ROOT = W / "data" / "countix_qa"
FRAMES = W / "data" / "frames_countix"
CLIPS = ROOT / "clips"
OUT = W / "p0_count_capacity" / "countix_train_units.jsonl"
REPO = "kekeYeah/Countix-QA"


def fixed_grid(n, k=32):
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def main():
    parsed = json.loads((ROOT / "countix_parsed.json").read_text(encoding="utf-8"))
    rows = [r for r in parsed["train"] if r["count"] <= 4]
    print("train candidates:", len(rows), flush=True)
    zpath = Path(hf_hub_download(REPO, "cut_countix_train.zip", repo_type="dataset",
                                 endpoint="https://hf-mirror.com", local_dir=str(ROOT)))
    print("zip:", round(zpath.stat().st_size / 1024**2), "MB", flush=True)
    CLIPS.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)

    units = []
    done = 0
    with zipfile.ZipFile(zpath) as archive:
        names = {Path(n).name: n for n in archive.namelist() if n.endswith(".mp4")}
        print("clips in zip:", len(names), flush=True)
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
                frame_dir.mkdir(parents=True, exist_ok=True)
                for slot, index in enumerate(fixed_grid(len(frames))):
                    Image.fromarray(frames[index]).save(frame_dir / f"{slot:02d}.jpg", quality=95)
            units.append(dict(unit_id=f"{vid}|countix", video_id=vid, action="repeated action",
                              split="countix_train", count=int(row["count"]), tier="repeated",
                              balanced=False, num_frames=0,
                              question="How often is the repeated action performed in the video? Answer with a single integer.",
                              answer=str(row["count"])))
            done += 1
            if done % 100 == 0:
                print("done", done, flush=True)

    OUT.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in units), encoding="utf-8")
    from collections import Counter
    print("units:", len(units), Counter(u["count"] for u in units).most_common(), flush=True)


if __name__ == "__main__":
    main()
