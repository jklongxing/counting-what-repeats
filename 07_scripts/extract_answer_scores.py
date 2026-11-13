"""Extract per-answer scores for the count digits, for output-calibration baselines.

For each question we run one forward pass over the 32 images plus the prompt and record the
next-token logits for the digits 1-8. That vector is what an output calibration can shift;
it is not recoverable from a decoded answer alone.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

W = Path("/home/jiquan/video-state-evidence/repetition_readout_v1")
RUNS = W / "runs"
FRAMES = W / "data" / "frames"
UNITS = W / "p0_count_capacity" / "benchmark3_units.jsonl"
VAL = W / "p0_count_capacity" / "val_units.jsonl"


def pick_frames(paths, wanted):
    if wanted >= len(paths):
        return paths
    step = len(paths) / wanted
    return [paths[min(len(paths) - 1, int(i * step))] for i in range(wanted)]


def build_model(model_path, adapter, gpu):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=512 * 512, min_pixels=32 * 32 * 4)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=f"cuda:{gpu}", attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return processor, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--units", default=str(UNITS))
    ap.add_argument("--include-val", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", dest="num_shards", type=int, default=1)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    units = [json.loads(l) for l in Path(args.units).read_text(encoding="utf-8").splitlines()]
    if args.include_val:
        units += [json.loads(l) for l in VAL.read_text(encoding="utf-8").splitlines()]
    units = [u for u in units if (FRAMES / u["video_id"]).exists()
             and len(list((FRAMES / u["video_id"]).glob("*.jpg"))) >= 32]
    todo = units[args.shard::args.num_shards]
    print(f"units {len(units)} shard {args.shard}/{args.num_shards} todo {len(todo)}", flush=True)

    processor, model = build_model(args.model, args.adapter, args.gpu)
    digit_ids = []
    for d in range(1, 9):
        ids = processor.tokenizer.encode(str(d), add_special_tokens=False)
        digit_ids.append(ids[0] if len(ids) == 1 else ids[-1])
    print("digit ids", digit_ids, flush=True)

    out = RUNS / f"scores_{args.tag}_s{args.shard}.jsonl"
    start = time.monotonic()
    with out.open("w", encoding="utf-8") as handle:
        for index, unit in enumerate(todo, 1):
            paths = sorted((FRAMES / unit["video_id"]).glob("*.jpg"))
            chosen = pick_frames(paths, args.frames)
            images = [Image.open(p).convert("RGB") for p in chosen]
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": unit["question"]})
            text = processor.apply_chat_template([{"role": "user", "content": content}],
                                                 tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors="pt").to(f"cuda:{args.gpu}")
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1]
            scores = [float(logits[i].float()) for i in digit_ids]
            handle.write(json.dumps(dict(unit_id=unit["unit_id"], video_id=unit["video_id"],
                                         split=unit["split"], count=unit["count"],
                                         action=unit["action"], scores=scores), ensure_ascii=False) + "\n")
            handle.flush()
            if index % 50 == 0 or index == len(todo):
                print(f"{index}/{len(todo)} {time.monotonic()-start:.0f}s", flush=True)
    print("saved", out)


if __name__ == "__main__":
    main()
