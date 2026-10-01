"""
Тест калибровки на СИНТЕТИЧЕСКОМ изображении (без реальной игры).

Рисуем упрощённый стол пасьянса: зелёный фон + 7 белых прямоугольников
("карт") на известных нам заранее координатах, потом проверяем, что
calibrate() находит те же координаты сама, не зная их заранее.

Это не заменяет проверку на реальных скриншотах игры — но доказывает,
что сам алгоритм (пороги + кластеризация проекции) логически верен,
прежде чем тратить время на реальные скриншоты.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
from PIL import Image, ImageDraw

from vision.calibration import calibrate, detect_columns

TABLE_BG = (30, 100, 40)      # зелёное сукно
CARD_BG = (255, 255, 255)     # белая карта


def make_synthetic_table(
    width=1400, height=900,
    n_columns=7, card_w=140, card_h=190,
    gap=40, top_margin=80,
) -> tuple[np.ndarray, list[int]]:
    """Возвращает (изображение, ground_truth_x_centers)."""
    img = Image.new("RGB", (width, height), TABLE_BG)
    draw = ImageDraw.Draw(img)

    total_width = n_columns * card_w + (n_columns - 1) * gap
    start_x = (width - total_width) // 2

    ground_truth_centers = []
    for col in range(n_columns):
        x0 = start_x + col * (card_w + gap)
        x1 = x0 + card_w
        draw.rectangle([x0, top_margin, x1, top_margin + card_h], fill=CARD_BG)
        ground_truth_centers.append(x0 + card_w // 2)

    return np.array(img), ground_truth_centers


def test_detect_columns_matches_ground_truth():
    image, ground_truth = make_synthetic_table()
    detected = detect_columns(image, TABLE_BG, expected_columns=7)

    assert len(detected) == len(ground_truth), (
        f"Нашла {len(detected)} колонок вместо {len(ground_truth)}"
    )
    for gt, det in zip(sorted(ground_truth), sorted(detected)):
        assert abs(gt - det) <= 3, f"Ground truth {gt}, обнаружено {det} — расхождение слишком большое"


def test_calibrate_returns_correct_top_y():
    image, _ = make_synthetic_table(top_margin=80)
    geometry = calibrate(image, table_bg_rgb=TABLE_BG, expected_columns=7)
    assert abs(geometry.table_top_y - 80) <= 2


def test_calibrate_fails_gracefully_on_wrong_color():
    image, _ = make_synthetic_table()
    try:
        calibrate(image, table_bg_rgb=(0, 0, 0), expected_columns=7)  # чёрный фон стола -- в картинке его нет
        assert False, "Должна была выбросить ValueError"
    except ValueError as e:
        assert "table_bg_rgb" in str(e) or "Не нашла" in str(e)


if __name__ == "__main__":
    image, ground_truth = make_synthetic_table()
    geometry = calibrate(image, table_bg_rgb=TABLE_BG, expected_columns=7)

    print("Ground truth центры колонок:", sorted(ground_truth))
    print("Обнаруженные центры колонок:", sorted(geometry.column_x_centers))
    print("Обнаруженная ширина колонки:", geometry.column_width)
    print("Обнаруженный top_y:", geometry.table_top_y)

    # Сохраняем картинку, чтобы можно было посмотреть глазами.
    # (Windows-совместимый путь рядом с проектом, а не /tmp из Linux.)
    out_path = Path(__file__).resolve().parent.parent / "dataset" / "synthetic_table.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_path)
    print(f"\nКартинка сохранена: {out_path}")
