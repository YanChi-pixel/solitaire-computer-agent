"""
Calibrate the geometry of the solitaire table.

Goal: turn a screenshot into a set of pixel coordinates (where the columns
are, where the top of the stacks is) without ML. Classic CV (color thresholds
+ projections), because the image source is clean rendered graphics.

Robust to EMPTY columns: if fewer than 7 columns are visible, we return
however many we actually find (a column is a card-like cluster width ~50..220
px), instead of raising an error. The collector then crops corners from the
columns that do have cards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TableGeometry:
    column_x_centers: list[int]
    column_width: int
    table_top_y: int


def _non_background_mask(pixels, table_bg_rgb, tolerance=40):
    diff = np.abs(pixels.astype(int) - np.array(table_bg_rgb))
    return diff.sum(axis=-1) > tolerance * 3


def detect_columns(image, table_bg_rgb, expected_columns=7, search_row_y=None, tolerance=40):
    """
    Find x-centers of card-like clusters on a single horizontal row.
    A "card" cluster must be roughly a card width (>= 50 px, <= 220 px);
    this filters out wide shadows/background and thin noise.
    """
    h, w = image.shape[:2]
    if search_row_y is None:
        search_row_y = h // 3

    row = image[search_row_y, :, :3]
    mask_1d = _non_background_mask(row, table_bg_rgb, tolerance)

    clusters = []
    current = []
    for x, is_card in enumerate(mask_1d):
        if is_card:
            current.append(x)
        elif current:
            clusters.append(current)
            current = []
    if current:
        clusters.append(current)

    min_w = 50   # card is wider than this
    max_w = 220  # card is narrower than this (shadows can be wider)
    clusters = [c for c in clusters if min_w <= len(c) <= max_w]

    centers = sorted(int(np.mean(c)) for c in clusters)
    return centers


def find_tableau_row(image, table_bg_rgb, expected_columns=7, tolerance=40,
                     y_start_fraction=0.30, y_end_fraction=0.80, step=5):
    """
    Find a row in the LOWER (tableau) area with the most card-like clusters.
    """
    h = image.shape[0]
    y_from = int(h * y_start_fraction)
    y_to = int(h * y_end_fraction)

    best_y = None
    best_count = 0
    for y in range(y_from, y_to, step):
        centers = detect_columns(image, table_bg_rgb, expected_columns, search_row_y=y, tolerance=tolerance)
        if len(centers) >= expected_columns:
            return y
        if len(centers) > best_count:
            best_count = len(centers)
            best_y = y

    if best_y is not None and best_count >= 1:
        return best_y

    raise ValueError("No card columns found in the tableau area. Check table_bg_rgb color.")


def detect_table_top(image, table_bg_rgb, column_x, known_row_y, tolerance=40):
    column_pixels = image[:, column_x, :3]
    mask = _non_background_mask(column_pixels, table_bg_rgb, tolerance)
    y = known_row_y
    while y > 0 and mask[y - 1]:
        y -= 1
    return int(y)


def calibrate(image, table_bg_rgb, expected_columns=7, search_row_y=None, tolerance=40):
    if search_row_y is None:
        search_row_y = find_tableau_row(image, table_bg_rgb, expected_columns, tolerance)
    centers = detect_columns(image, table_bg_rgb, expected_columns, search_row_y, tolerance)

    if len(centers) == 0:
        raise ValueError("No card columns found. Check table_bg_rgb color.")

    column_width = int(np.mean(np.diff(centers))) if len(centers) > 1 else 130

    top_y = min(
        detect_table_top(image, table_bg_rgb, x, known_row_y=search_row_y, tolerance=tolerance)
        for x in centers
    )

    return TableGeometry(column_x_centers=centers, column_width=column_width, table_top_y=top_y)
