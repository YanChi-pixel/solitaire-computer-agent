"""Пиксельный разбор колонок: где реально начинаются/кончаются карты."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from vision import screen_to_board as s2b

img = np.array(Image.open(ROOT / "_live_frame.png").convert("RGB"))
H, W = img.shape[:2]
centers = s2b.find_tableau_columns(img)
card_w, card_h = s2b.measure_card_size(img, centers)
print(f"centers={centers} card={card_w}x{card_h}")


def classify(px):
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    if g > r + 20 and g > b + 20:
        return "G"          # зелёное сукно
    if b > r + 15 and b > g:
        return "B"          # синяя рубашка
    if r > 200 and g > 200 and b > 200:
        return "W"          # белый фон карты
    if r < 70 and g < 70 and b < 70:
        return "."          # тёмное (символ/текст)
    if abs(r - g) < 25 and abs(g - b) < 25 and r > 110:
        return "#"          # серо-белое (край карты)
    return "o"              # красное / прочее


print("\n=== вертикальные профили (y 300..960), x = центр и x = край ===")
for i, cx in enumerate(centers):
    for label, x in (("center", cx), ("edge", cx - card_w // 2 + 6)):
        runs = []
        prev = None
        start = None
        for y in range(300, 960):
            c = classify(img[y, x])
            if c != prev:
                if prev is not None:
                    runs.append((prev, start, y - 1, y - start))
                prev = c
                start = y
        runs.append((prev, start, 959, 960 - start))
        # печатаем только значимые прогоны
        sig = [r for r in runs if r[3] >= 3]
        compact = " ".join(f"{c}{a}-{b}({n})" for c, a, b, n in sig)
        print(f"[{i+1}] cx={cx} {label:6s} x={x:4d}: {compact}")

# ── сохраню увеличенные вырезки колонок 1 и 4 ────────────────────────
for idx in (0, 3):
    cx = centers[idx]
    y0, y1 = 300, 720
    x0 = max(0, cx - card_w // 2 - 20)
    x1 = min(W, cx + card_w // 2 + 20)
    crop = img[y0:y1, x0:x1]
    Image.fromarray(crop).resize((crop.shape[1] * 2, crop.shape[0] * 2), Image.LANCZOS).save(
        ROOT / f"_col{idx+1}.png")
    print(f"saved _col{idx+1}.png")
