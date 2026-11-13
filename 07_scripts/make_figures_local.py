"""Local figure pipeline: reads data/plot_data.json, writes figures/*.png.

Runs entirely offline. Figure 6 (the qualitative example) is not regenerated here,
because it needs the source frames on the server; the existing PNG is kept.

Usage: python make_figures_local.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data" / "plot_data.json").read_text(encoding="utf-8"))
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

CANON = DATA["canonical"]
SCORES = CANON["scores"]
PRIOR = CANON["prior_score"]
BLUE = "#2f5f8f"
RED = "#a8452f"
GREEN = "#3f7a52"
GREY = "#8a8a8a"


def fig1():
    dist = DATA["count_dist"]
    splits = DATA["splits"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    labels = ["1", "2", "3", "4+"]
    values = [dist[k] for k in ("1", "2", "3", "4")]
    axes[0].bar(labels, values, color=BLUE)
    axes[0].set_xlabel("occurrences in a video")
    axes[0].set_ylabel("questions")
    axes[0].set_title(f"count distribution ({DATA['total']:,} questions)")
    for x, v in zip(labels, values):
        axes[0].text(x, v + 30, str(v), ha="center", fontsize=8)
    order = ["train", "dev", "test"]
    axes[1].bar(order, [splits[o] for o in order], color=GREEN)
    axes[1].set_ylabel("questions")
    axes[1].set_title("video-level splits (evaluated)")
    for i, o in enumerate(order):
        axes[1].text(i, splits[o] + 40, str(splits[o]), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_benchmark.png", dpi=200)
    plt.close(fig)


def fig2():
    curves = DATA["frame_curve"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    colors = {"Qwen3-VL-8B": BLUE, "Qwen2.5-VL-7B": RED, "InternVL3.5-8B": GREEN}
    for model, points in curves.items():
        xs = sorted(int(k) for k in points)
        ys = [points[str(x)] for x in xs]
        ax.plot(xs, ys, marker="o", label=model, color=colors.get(model))
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 128, 256])
    ax.set_xticklabels(["4", "8", "16", "32", "128", "256"])
    ax.set_xlabel("frames per video")
    ax.set_ylabel("repeated-action accuracy (%)")
    ax.set_title("repetition accuracy vs frame budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_frames.png", dpi=200)
    plt.close(fig)


def fig3():
    rows = [("test zero-shot", "zero_shot_test"), ("test LoRA", "lora_test"),
            ("VAL zero-shot", "val_base"), ("VAL LoRA", "val_lora")]
    names, once, rep = [], [], []
    for label, key in rows:
        v = SCORES.get(key)
        if not v:
            continue
        names.append(label)
        once.append(v["once"]["acc"] * 100)
        rep.append(v["repeated"]["exact4"] * 100)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - 0.2, once, 0.4, label="single occurrence", color=BLUE)
    ax.bar(x + 0.2, rep, 0.4, label="repeated", color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("supervision moves singles, not repetitions")
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(once, rep)):
        ax.text(i - 0.2, a + 1, f"{a:.0f}", ha="center", fontsize=7)
        ax.text(i + 0.2, b + 1, f"{b:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_supervision.png", dpi=200)
    plt.close(fig)


def fig4():
    multi = {row["adapter"]: row for row in DATA["multiseed_probe"]}
    adapters = [("seed 0", "lora_test", "seed0"), ("seed 7", "seed7_test", "seed7"),
                ("seed 8", "seed8_test", "seed8")]
    names, text, head, sd = [], [], [], []
    for label, key, mkey in adapters:
        v = SCORES.get(key)
        row = multi.get(mkey)
        if not v or not row:
            continue
        names.append(label)
        text.append(v["repeated"]["exact4"] * 100)
        head.append(row["head_repeated"] * 100)
        sd.append(row["head_repeated_sd"] * 100)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(x - 0.2, text, 0.4, label="text output", color=RED)
    ax.bar(x + 0.2, head, 0.4, yerr=sd, capsize=3, label="readout", color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("repeated accuracy (%)")
    ax.set_title("matched on adapter: readout helps every seed")
    ax.set_ylim(0, 75)
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(text, head)):
        ax.text(i - 0.2, a + 1, f"{a:.0f}", ha="center", fontsize=7)
        ax.text(i + 0.2, b + 2, f"{b:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_readout.png", dpi=200)
    plt.close(fig)


def fig5():
    entries = [("zero-shot", "zero_shot_test"), ("LoRA", "lora_test"),
               ("full fine-tuning", "fullft_test"), ("class-weighted LoRA", "balanced_lora_test"),
               ("readout", "head_test")]
    names, values = [], []
    for label, key in entries:
        v = SCORES.get(key)
        if v:
            names.append(label)
            values.append(v["repeated"]["exact4"] * 100)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(names, values, color=BLUE)
    ax.set_xlabel("repeated-action accuracy (%)")
    ax.set_title("methods on the repeated subset")
    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlim(0, 70)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_methods.png", dpi=200)
    plt.close(fig)


def fig7():
    report = DATA["error_structure"]["by_count"]
    counts = ["1", "2", "3", "4"]
    series = {"zero-shot": "zero-shot", "LoRA": "LoRA", "readout": "head"}
    x = np.arange(len(counts))
    widths = 0.26
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, (label, key) in enumerate(series.items()):
        vals = [report[key][c][1] * 100 for c in counts]
        ax.bar(x + (k - 1) * widths, vals, widths, label=label)
        for i, v in enumerate(vals):
            ax.text(x[i] + (k - 1) * widths, v + 1, f"{v:.0f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(["1", "2", "3", "4+"])
    ax.set_xlabel("true count")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("where the errors are")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_by_count.png", dpi=200)
    plt.close(fig)


def fig8():
    linear = DATA["head_ablation"]["linear"]
    points = {
        "zero-shot Qwen2.5": (SCORES["qwen25_test"]["once"]["acc"] * 100,
                              SCORES["qwen25_test"]["repeated"]["exact4"] * 100),
        "zero-shot Qwen3": (SCORES["zero_shot_test"]["once"]["acc"] * 100,
                            SCORES["zero_shot_test"]["repeated"]["exact4"] * 100),
        "zero-shot InternVL3.5": (SCORES["internvl_test"]["once"]["acc"] * 100,
                                  SCORES["internvl_test"]["repeated"]["exact4"] * 100),
        "per-action prior": (PRIOR["once"]["acc"] * 100, PRIOR["repeated"]["exact4"] * 100),
        "LoRA": (SCORES["lora_test"]["once"]["acc"] * 100, SCORES["lora_test"]["repeated"]["exact4"] * 100),
        "full fine-tuning": (SCORES["fullft_test"]["once"]["acc"] * 100,
                             SCORES["fullft_test"]["repeated"]["exact4"] * 100),
        "class-weighted LoRA": (SCORES["balanced_lora_test"]["once"]["acc"] * 100,
                                SCORES["balanced_lora_test"]["repeated"]["exact4"] * 100),
        "MLP readout": (SCORES["head_test"]["once"]["acc"] * 100,
                        SCORES["head_test"]["repeated"]["exact4"] * 100),
        "logistic readout": (linear["once"] * 100, linear["repeated"] * 100),
    }
    names = list(points)
    frontier = [n for n in names
                if not any(m != n and points[m][0] >= points[n][0] and points[m][1] >= points[n][1]
                           for m in names)]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, (x, y) in points.items():
        on_front = name in frontier
        ax.scatter(x, y, s=70, zorder=3,
                   color=BLUE if on_front else GREY,
                   edgecolor="black" if on_front else "none",
                   marker="o" if on_front else "x")
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 5), fontsize=7)

    stair = sorted((points[n] for n in frontier), key=lambda p: p[0])
    ax.plot([p[0] for p in stair], [p[1] for p in stair], color=BLUE, alpha=0.5,
            linestyle="--", zorder=2, label="Pareto frontier")
    ax.set_xlabel("single-occurrence accuracy (%)")
    ax.set_ylabel("repeated-action accuracy (%)")
    ax.set_title("one trade-off frontier, no free improvement")
    ax.set_xlim(40, 105)
    ax.set_ylim(10, 65)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig8_tradeoff.png", dpi=200)
    plt.close(fig)
    print("frontier:", frontier)


def main():
    fig1(); fig2(); fig3(); fig4(); fig5(); fig7(); fig8()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
