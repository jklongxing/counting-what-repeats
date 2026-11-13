"""Graphical abstract for the JVCI submission.

Elsevier asks for at least 531 x 1328 pixels (h x w) at 300 dpi, readable fonts and few
words. This draws 1328 x 531 with a four-step flow.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "figures" / "graphical_abstract.png"
W, H = 1328, 531
BLUE = (47, 95, 143)
RED = (168, 69, 47)
GREY = (110, 110, 110)
LIGHT = (238, 242, 247)
INK = (25, 25, 25)


def font(size, bold=False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), ("segoeuib.ttf" if bold else "segoeui.ttf")):
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def main():
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)

    draw.text((W // 2, 34), "Counting repeated actions in natural video", font=font(30, True), fill=INK, anchor="mm")

    boxes = [
        ("3,710 questions\n1,563 videos", LIGHT, INK),
        ("zero-shot 49.2%\naction prior 72.6%", LIGHT, INK),
        ("supervision\n+39 single\n+1 repeated", LIGHT, INK),
        ("readout\n+12.9 repeated", (222, 233, 244), INK),
    ]
    box_w, box_h, gap = 268, 186, 36
    total = len(boxes) * box_w + (len(boxes) - 1) * gap
    x = (W - total) // 2
    y = 92
    for text, colour, ink in boxes:
        draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=14, fill=colour, outline=BLUE, width=3)
        lines = text.split("\n")
        line_h = 40
        start = y + box_h // 2 - (len(lines) - 1) * line_h // 2
        for i, line in enumerate(lines):
            size = 27 if i == 0 else 24
            weight = i == 0
            draw.text((x + box_w // 2, start + i * line_h), line, font=font(size, weight), fill=ink, anchor="mm")
        x += box_w + gap

    # arrows
    x = (W - total) // 2 + box_w
    for _ in range(len(boxes) - 1):
        cy = y + box_h // 2
        draw.line([(x + 6, cy), (x + gap - 8, cy)], fill=GREY, width=4)
        draw.polygon([(x + gap - 8, cy), (x + gap - 20, cy - 8), (x + gap - 20, cy + 8)], fill=GREY)
        x += box_w + gap

    draw.text((W // 2, 330), "Same video, two actions, different counts (272 pairs)", font=font(25, True), fill=INK, anchor="mm")

    bars = [("readout", 36.8, BLUE), ("LoRA", 28.7, GREY), ("action prior", 18.0, RED)]
    bx, by, bw, bh = 250, 360, 320, 26
    for i, (label, value, colour) in enumerate(bars):
        yy = by + i * 42
        draw.text((bx - 16, yy + bh // 2), label, font=font(21), fill=INK, anchor="rm")
        width = int(bw * value / 40.0)
        draw.rectangle([bx, yy, bx + width, yy + bh], fill=colour)
        draw.text((bx + width + 12, yy + bh // 2), f"{value:.1f}%", font=font(21, True), fill=INK, anchor="lm")

    draw.text((W // 2, 505), "The readout uses the video and the asked action, but corrects answers rather than counting.",
              font=font(23), fill=INK, anchor="mm")

    image.save(OUT, dpi=(300, 300))
    print("wrote", OUT, image.size)


if __name__ == "__main__":
    main()
