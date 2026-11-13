"""Draw every data figure with Pillow, so no extra library is needed.

Reads data/plot_data.json and writes figures/*.png. Figure 6 (the qualitative
example) is left untouched because it needs the source frames on the server.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data" / "plot_data.json").read_text(encoding="utf-8"))
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

CANON = DATA["canonical"]
SCORES = CANON["scores"]
PRIOR = CANON["prior_score"]

W, H = 1800, 1080
BLUE = (47, 95, 143)
RED = (168, 69, 47)
GREEN = (63, 122, 82)
GREY = (138, 138, 138)
INK = (25, 25, 25)
GRID = (225, 225, 225)


def font(size: int, bold: bool = False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), ("segoeuib.ttf" if bold else "segoeui.ttf")):
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


class Chart:
    def __init__(self, width=W, height=H, left=156, right=72, top=108, bottom=132):
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.box = (left, top, width - right, height - bottom)
        self.width, self.height = width, height

    def title(self, text):
        self.draw.text((self.width // 2, 48), text, font=font(36, True), fill=INK, anchor="mm")

    def axes(self, xlim, ylim, xticks, yticks, xlabel="", ylabel="", yfmt="{:.0f}"):
        x0, y0, x1, y1 = self.box
        self.xlim, self.ylim = xlim, ylim
        for value in yticks:
            y = self.map_y(value)
            self.draw.line([(x0, y), (x1, y)], fill=GRID, width=1)
            self.draw.text((x0 - 12, y), yfmt.format(value), font=font(21), fill=INK, anchor="rm")
        for value in xticks:
            x = self.map_x(value)
            self.draw.text((x, y1 + 17), str(value), font=font(21), fill=INK, anchor="mt")
        self.draw.line([(x0, y1), (x1, y1)], fill=INK, width=2)
        self.draw.line([(x0, y0), (x0, y1)], fill=INK, width=2)
        if xlabel:
            self.draw.text(((x0 + x1) // 2, self.height - 38), xlabel, font=font(22), fill=INK, anchor="mm")
        if ylabel:
            self.rotated_label(ylabel, (28, (y0 + y1) // 2))

    def rotated_label(self, text, centre):
        probe = Image.new("RGBA", (10, 10))
        drawer = ImageDraw.Draw(probe)
        f = font(20)
        box = drawer.textbbox((0, 0), text, font=f)
        tile = Image.new("RGBA", (box[2] - box[0] + 8, box[3] - box[1] + 8), (255, 255, 255, 0))
        ImageDraw.Draw(tile).text((4, 4), text, font=f, fill=INK)
        tile = tile.rotate(90, expand=True)
        self.image.paste(tile, (int(centre[0] - tile.width / 2), int(centre[1] - tile.height / 2)), tile)

    def map_x(self, value):
        x0, _, x1, _ = self.box
        lo, hi = self.xlim
        return x0 + (value - lo) / (hi - lo) * (x1 - x0)

    def map_y(self, value):
        _, y0, _, y1 = self.box
        lo, hi = self.ylim
        return y1 - (value - lo) / (hi - lo) * (y1 - y0)

    def bar(self, center, value, width, color, label=None):
        x = self.map_x(center)
        top = self.map_y(value)
        _, _, _, base = self.box
        self.draw.rectangle([x - width / 2, top, x + width / 2, base], fill=color, outline="white")
        if label:
            self.draw.text((x, top - 10), label, font=font(16), fill=INK, anchor="mb")

    def hbar(self, y_index, count, value, color, label=None):
        _, y0, x1, y1 = self.box
        step = (y1 - y0) / count
        y = y0 + step * (y_index + 0.5)
        x = self.map_x(value)
        x0 = self.map_x(0)
        self.draw.rectangle([x0, y - step * 0.32, x, y + step * 0.32], fill=color)
        if label:
            self.draw.text((x + 10, y), label, font=font(17), fill=INK, anchor="lm")

    def line(self, xs, ys, color, marker_size=9):
        pts = [(self.map_x(x), self.map_y(y)) for x, y in zip(xs, ys)]
        self.draw.line(pts, fill=color, width=4, joint="curve")
        for x, y in pts:
            self.draw.ellipse([x - marker_size, y - marker_size, x + marker_size, y + marker_size],
                              fill=color, outline="white", width=2)

    def dot(self, x, y, color, label=None, filled=True, dx=12, dy=-14):
        px, py = self.map_x(x), self.map_y(y)
        if filled:
            self.draw.ellipse([px - 10, py - 10, px + 10, py + 10], fill=color, outline=INK, width=2)
        else:
            self.draw.line([(px - 8, py - 8), (px + 8, py + 8)], fill=color, width=3)
            self.draw.line([(px - 8, py + 8), (px + 8, py - 8)], fill=color, width=3)
        if label:
            self.draw.text((px + dx, py + dy), label, font=font(17), fill=INK, anchor="lb")

    def legend(self, entries, x=None, y=None):
        x = x or (self.box[0] + 30)
        y = y or (self.box[1] + 30)
        for i, (label, color, kind) in enumerate(entries):
            yy = y + i * 32
            if kind == "bar":
                self.draw.rectangle([x, yy - 10, x + 26, yy + 10], fill=color)
            elif kind == "line":
                self.draw.line([(x, yy), (x + 26, yy)], fill=color, width=4)
                self.draw.ellipse([x + 8, yy - 6, x + 20, yy + 6], fill=color)
            self.draw.text((x + 36, yy), label, font=font(18), fill=INK, anchor="lm")

    def save(self, name):
        self.image.save(OUT / name)
        print("wrote", name)


def fig1():
    dist = DATA["count_dist"]
    splits = DATA["splits"]
    c = Chart(left=120, right=70)
    c.title(f"count distribution ({DATA['total']:,} questions)")
    c.axes((0.5, 4.5), (0, 2900), [], [0, 500, 1000, 1500, 2000, 2500],
           "occurrences in a video", "questions")
    labels = ["1", "2", "3", "4+"]
    for i, key in enumerate(["1", "2", "3", "4"]):
        c.bar(i + 1, dist[key], 110, BLUE, str(dist[key]))
    for i, lab in enumerate(labels):
        x = c.map_x(i + 1)
        c.draw.text((x, c.box[3] + 14), lab, font=font(18), fill=INK, anchor="mt")
    c.save("fig1_counts.png")

    c2 = Chart(left=120, right=120)
    c2.title("video-level splits (evaluated)")
    c2.axes((0.5, 3.5), (0, 2900), [], [0, 500, 1000, 1500, 2000, 2500], "", "questions")
    for i, key in enumerate(["train", "dev", "test"]):
        c2.bar(i + 1, splits[key], 150, GREEN, str(splits[key]))
        x = c2.map_x(i + 1)
        c2.draw.text((x, c2.box[3] + 14), key, font=font(18), fill=INK, anchor="mt")
    c2.save("fig1_splits.png")

    paired = Image.new("RGB", (W * 2, H), "white")
    paired.paste(Image.open(OUT / "fig1_counts.png"), (0, 0))
    paired.paste(Image.open(OUT / "fig1_splits.png"), (W, 0))
    paired.save(OUT / "fig1_benchmark.png")
    print("wrote fig1_benchmark.png")


def fig2():
    curves = DATA["frame_curve"]
    c = Chart()
    c.title("repetition accuracy vs frame budget")
    c.axes((math.log2(4) - 0.35, math.log2(256) + 0.35), (0, 45), [], [0, 10, 20, 30, 40],
           "frames per video (log scale)", "repeated accuracy (%)")
    for tick in (4, 8, 16, 32, 128, 256):
        x = c.map_x(math.log2(tick))
        c.draw.text((x, c.box[3] + 14), str(tick), font=font(18), fill=INK, anchor="mt")
    colors = {"Qwen3-VL-8B": BLUE, "Qwen2.5-VL-7B": RED, "InternVL3.5-8B": GREEN}
    for model, points in curves.items():
        xs = sorted(int(k) for k in points)
        c.line([math.log2(x) for x in xs], [points[str(x)] for x in xs], colors[model])
    c.legend([(m, colors[m], "line") for m in curves], x=c.box[0] + 50, y=c.box[1] + 30)
    c.save("fig2_frames.png")


def fig3():
    rows = [("test zero-shot", "zero_shot_test"), ("test LoRA", "lora_test"),
            ("VAL zero-shot", "val_base"), ("VAL LoRA", "val_lora")]
    c = Chart()
    c.title("supervision moves singles, not repetitions")
    c.axes((0.5, 4.5), (0, 105), [], [0, 20, 40, 60, 80, 100], "", "accuracy (%)")
    for i, (label, key) in enumerate(rows):
        v = SCORES[key]
        c.bar(i + 1 - 0.18, v["once"]["acc"] * 100, 95, BLUE, f"{v['once']['acc']*100:.0f}")
        c.bar(i + 1 + 0.18, v["repeated"]["exact4"] * 100, 95, RED, f"{v['repeated']['exact4']*100:.0f}")
        x = c.map_x(i + 1)
        c.draw.text((x, c.box[3] + 16), label, font=font(17), fill=INK, anchor="mt")
    c.legend([("single occurrence", BLUE, "bar"), ("repeated", RED, "bar")],
             x=c.box[0] + 40, y=c.box[1] + 40)
    c.save("fig3_supervision.png")


def fig4():
    multi = {row["adapter"]: row for row in DATA["multiseed_probe"]}
    rows = [("seed 0", "lora_test", "seed0"), ("seed 7", "seed7_test", "seed7"),
            ("seed 8", "seed8_test", "seed8")]
    c = Chart()
    c.title("matched on adapter: readout helps every seed")
    c.axes((0.5, 3.5), (0, 75), [], [0, 10, 20, 30, 40, 50, 60, 70], "", "repeated accuracy (%)")
    for i, (label, key, mkey) in enumerate(rows):
        text = SCORES[key]["repeated"]["exact4"] * 100
        row = multi[mkey]
        head = row["head_repeated"] * 100
        sd = row["head_repeated_sd"] * 100
        c.bar(i + 1 - 0.18, text, 95, RED, f"{text:.0f}")
        c.bar(i + 1 + 0.18, head, 95, BLUE, f"{head:.0f}")
        x = c.map_x(i + 1 + 0.18)
        top = c.map_y(head + sd)
        bottom = c.map_y(max(head - sd, 0))
        c.draw.line([(x, top), (x, bottom)], fill=INK, width=2)
        c.draw.line([(x - 8, top), (x + 8, top)], fill=INK, width=2)
        c.draw.line([(x - 8, bottom), (x + 8, bottom)], fill=INK, width=2)
        c.draw.text((c.map_x(i + 1), c.box[3] + 16), label, font=font(18), fill=INK, anchor="mt")
    c.legend([("text output", RED, "bar"), ("readout", BLUE, "bar")], x=c.box[0] + 40, y=c.box[1] + 40)
    c.save("fig4_readout.png")


def fig5():
    entries = [("zero-shot", "zero_shot_test"), ("LoRA", "lora_test"),
               ("full fine-tuning", "fullft_test"), ("class-weighted LoRA", "balanced_lora_test"),
               ("readout", "head_test")]
    c = Chart(left=300, right=120)
    c.title("methods on the repeated subset")
    c.axes((0, 70), (0, len(entries)), [0, 10, 20, 30, 40, 50, 60, 70], [],
           "repeated-action accuracy (%)", "")
    for i, (label, key) in enumerate(entries):
        value = SCORES[key]["repeated"]["exact4"] * 100
        c.hbar(i, len(entries), value, BLUE, f"{value:.1f}")
        y = c.box[1] + (c.box[3] - c.box[1]) / len(entries) * (i + 0.5)
        c.draw.text((c.box[0] - 14, y), label, font=font(21), fill=INK, anchor="rm")
    c.save("fig5_methods.png")


def fig7():
    report = DATA["error_structure"]["by_count"]
    c = Chart()
    c.title("where the errors are")
    c.axes((0.5, 4.5), (0, 105), [], [0, 20, 40, 60, 80, 100],
           "true count (4+ pooled)", "accuracy (%)")
    series = [("zero-shot", "zero-shot", GREY), ("LoRA", "LoRA", RED), ("readout", "head", BLUE)]
    for i, (label, key, color) in enumerate(series):
        for j, count in enumerate(["1", "2", "3", "4"]):
            value = report[key][count][1] * 100
            c.bar(j + 1 + (i - 1) * 0.25, value, 78, color, f"{value:.0f}")
    for j, lab in enumerate(["1", "2", "3", "4+"]):
        c.draw.text((c.map_x(j + 1), c.box[3] + 16), lab, font=font(18), fill=INK, anchor="mt")
    c.legend([(l, col, "bar") for l, _, col in series], x=c.box[2] - 420, y=c.box[1] + 30)
    c.save("fig7_by_count.png")


def fig8():
    """Single-occurrence against repeated accuracy for all evaluated configurations."""
    import json as _json
    summary = _json.loads((Path(__file__).resolve().parent / "DATA_BUNDLE_20261111" / "04_summaries" / "plane_points.json").read_text(encoding="utf-8"))
    points = {k: (v[0], v[1]) for k, v in summary.items()}
    points["zero-shot InternVL3.5"] = (66.7, 39.2)
    names = list(points)
    frontier = [n for n in names
                if not any(m != n and points[m][0] >= points[n][0] and points[m][1] >= points[n][1]
                           for m in names)]
    c = Chart(left=150, bottom=120)
    c.title("Single-occurrence and repeated-action accuracy of evaluated configurations")
    c.axes((45, 105), (10, 65), [50, 60, 70, 80, 90, 100], [10, 20, 30, 40, 50, 60],
           "single-occurrence accuracy (%)", "repeated accuracy (%)")
    stair = sorted((points[n] for n in frontier), key=lambda q: q[0])
    c.draw.line([(c.map_x(x), c.map_y(y)) for x, y in stair], fill=(150, 150, 200), width=3)
    offsets = {
        "zero-shot Qwen2.5": (14, -22), "zero-shot Qwen3": (14, 6),
        "zero-shot InternVL3.5": (14, -20), "per-action prior": (-14, 10),
        "LoRA": (-16, 10), "full fine-tuning": (16, 8),
        "class-weighted LoRA": (16, -4), "MLP readout": (14, -24),
        "logistic readout": (16, 12), "per-class digit bias": (-16, 12),
        "dev-matched repetition bias": (16, -18),
        "action-rule relabelling (test single 88.2, repeated 50.3)": (16, 4),
    }
    for name, (x, y) in points.items():
        on_front = name in frontier
        c.dot(x, y, BLUE if on_front else GREY, None, filled=on_front)
        dx, dy = offsets.get(name, (12, -14))
        px, py = c.map_x(x), c.map_y(y)
        label = name.replace("action-rule relabelling (test single 88.2, repeated 50.3)", "action-rule relabelling")
        c.draw.text((px + dx, py + dy), label, font=font(17), fill=INK,
                    anchor="rb" if dx < 0 else "lb")
    c.draw.text((c.box[0] + 30, c.box[1] + 30),
                "filled = Pareto-optimal within the plotted set, x = dominated",
                font=font(18), fill=INK, anchor="lm")
    c.save("fig8_tradeoff.png")
    print("frontier:", sorted(frontier))


def main():
    fig1(); fig2(); fig3(); fig4(); fig5(); fig7(); fig8()
    for tmp in ("fig1_counts.png", "fig1_splits.png"):
        path = OUT / tmp
        if path.exists():
            path.unlink()
    print("done")


if __name__ == "__main__":
    main()




