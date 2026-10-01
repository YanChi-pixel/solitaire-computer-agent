"""
Extract the corner of a face-up card — a cropped piece with the rank and
suit, exactly what goes into the classifier (MobileNet).

Found empirically on real screenshots: the back of a face-down card is a
saturated blue (~25, 95, 189), the face is almost white (~250+, 250+, 250+).
These colors are far enough from each other and from the green felt to tell
them apart with a simple threshold, without any ML.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision.calibration import TableGeometry, _non_background_mask


@dataclass
class CardCorner:
    column_index: int
    x: int
    y: int
    width: int
    height: int
    crop: np.ndarray  # сам вырезанный кусок картинки (H x W x 3)


def _is_front_face(pixel: np.ndarray, white_threshold: int = 200, gray_tolerance: int = 25) -> bool:
    """Похоже ли на белый/светлый фон лицевой стороны карты (не на синюю рубашку)."""
    r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
    is_bright = min(r, g, b) > white_threshold
    is_grayish = (max(r, g, b) - min(r, g, b)) < gray_tolerance
    return is_bright and is_grayish


def find_front_face_top(
    image: np.ndarray,
    column_x: int,
    search_from_y: int,
    max_search_height: int = 500,
    min_consecutive_rows: int = 6,
) -> int | None:
    """
    Находит y, где начинается ОТКРЫТАЯ (лицевая) часть карты в колонке.

    ВАЖНО: рубашка закрытой карты — не сплошной синий, а узор с белыми
    декоративными линиями. Одиночный белый пиксель может быть частью
    этого узора, а не настоящим началом лицевой стороны. Поэтому ищем
    не первую белую строку, а первую строку, после которой идёт ещё
    min_consecutive_rows подряд белых строк — узор рубашки такой длинной
    сплошной белой полосы не даёт, а сплошной белый фон лицевой карты даёт.
    """
    h = image.shape[0]
    y_end = min(search_from_y + max_search_height, h)

    consecutive = 0
    candidate_start = None
    for y in range(search_from_y, y_end):
        if _is_front_face(image[y, column_x]):
            if consecutive == 0:
                candidate_start = y
            consecutive += 1
            if consecutive >= min_consecutive_rows:
                return candidate_start
        else:
            consecutive = 0
            candidate_start = None
    return None


def _column_left_right(
    image: np.ndarray,
    table_bg_rgb: tuple[int, int, int],
    search_row_y: int,
    column_x_center: int,
    max_half_width: int,
    tolerance: int = 40,
) -> tuple[int, int]:
    """Находит левую и правую границу карты в колонке на заданной строке."""
    row = image[search_row_y, :, :3]
    mask = _non_background_mask(row, table_bg_rgb, tolerance)

    left = column_x_center
    while left > column_x_center - max_half_width and mask[left - 1]:
        left -= 1

    right = column_x_center
    w = image.shape[1]
    while right < min(column_x_center + max_half_width, w - 1) and mask[right + 1]:
        right += 1

    return left, right


def extract_visible_corners(
    image: np.ndarray,
    geometry: TableGeometry,
    table_bg_rgb: tuple[int, int, int],
    corner_width_ratio: float = 0.32,
    corner_height_ratio: float = 0.32,
) -> list[CardCorner]:
    """
    Для каждой колонки таблицы находит открытую (лицевую) карту и
    вырезает её верхний левый уголок — область с рангом и мастью,
    то есть то, что реально нужно классификатору.

    Колонки без открытой карты (пустые, или найти не удалось)
    пропускаются — вызывающий код должен сам решить, что с этим делать
    (например, пустая колонка допустима и это не ошибка).
    """
    corners: list[CardCorner] = []
    max_half_width = geometry.column_width // 2

    for col_idx, center_x in enumerate(geometry.column_x_centers):
        front_top = find_front_face_top(image, center_x, geometry.table_top_y)
        if front_top is None:
            continue  # нет открытой карты в этой колонке (пусто либо все закрыты)

        left, right = _column_left_right(
            image, table_bg_rgb, front_top + 5, center_x, max_half_width
        )
        card_width_px = right - left

        corner_w = max(int(card_width_px * corner_width_ratio), 10)
        corner_h = max(int(card_width_px * corner_height_ratio * 1.3), 10)  # уголок выше, чем шире

        crop = image[front_top:front_top + corner_h, left:left + corner_w].copy()
        corners.append(
            CardCorner(
                column_index=col_idx, x=left, y=front_top,
                width=corner_w, height=corner_h, crop=crop,
            )
        )

    return corners
