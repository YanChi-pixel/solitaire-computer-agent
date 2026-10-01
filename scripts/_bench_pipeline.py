"""Честные замеры пайплайна: захват, раскладка, чтение стола, распознавание.

Запуск: python scripts/_bench_pipeline.py
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from act import controller as act
from act.layout import Layout
from vision import screen_to_board as s2b
from vision.recognizer import get_recognizer

FRAME = ROOT / "tests" / "data" / "live_frame_2026-10-01.png"
img = np.array(Image.open(FRAME).convert("RGB"))
print(f"кадр {img.shape[1]}x{img.shape[0]}")

rec = get_recognizer()
centers = s2b.find_tableau_columns(img)
card_w, card_h = s2b.measure_card_size(img, centers)


def bench(name, fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"{name:44s} {dt:8.1f} мс")
    return dt


print("\n--- на сохранённом кадре ---")
t_find = bench("find_tableau_columns", lambda: s2b.find_tableau_columns(img), n=20)
t_size = bench("measure_card_size", lambda: s2b.measure_card_size(img, centers), n=20)
t_layout = bench("Layout(image)  (сетка + якоря ряда)", lambda: Layout(img), n=15)
t_s2b = bench("screen_to_board(image)  (полное чтение)", lambda: s2b.screen_to_board(img, rec), n=10)
t_one = bench("recognize_at_with_confidence  (1 карта)",
              lambda: rec.recognize_at_with_confidence(img, centers[0], 514, card_w, card_h), n=30)

print("\n--- на живом окне (если игра открыта) ---")
try:
    hwnd = act.find_game_window()
    t_cap = bench("capture_window(hwnd)", lambda: act.capture_window(hwnd), n=20)

    def full_cycle():
        image = act.capture_window(hwnd)
        board = s2b.screen_to_board(image, rec)
        return board

    t_full = bench("ПОЛНЫЙ ЦИКЛ: захват + чтение стола", full_cycle, n=10, warm=2)
    print(f"\nИТОГО полный цикл чтения стола: {t_full:.0f} мс "
          f"(захват {t_cap:.0f} + распознавание/геометрия {t_full - t_cap:.0f})")
except Exception as e:
    print(f"игра недоступна: {e}")
    t_cap = None

print(f"\nСПРАВКА: раскладка {t_layout:.0f} мс + чтение доски {t_s2b:.0f} мс = {t_layout + t_s2b:.0f} мс "
      f"(из них распознавание ~{t_one * 12:.0f} мс на ~12 карт)")
