"""
Table layout in pixel coordinates — for the Act layer.

Given a screenshot, determine:
- the centers of the 7 tableau columns and the y of each column's face-up card;
- the position of the stock, waste and 4 foundation piles.

This is needed to translate a logical Move ("move from column 2 to 5")
into real pixel coordinates for click_screen / drag_screen.
"""

from __future__ import annotations

import numpy as np

from vision.screen_to_board import (
    count_cards_in_column,
    find_card_top_y,
    find_stack_top,
    find_tableau_columns,
    measure_card_size,
)


def _largest_blue_run(blue_zone, W, min_width=60):
    """
    Find the leftmost solid blue x-run in the top zone (the stock pile).

    A "solid" column has many blue rows (>= 5), so thin UI lines and isolated
    pixels are ignored. Returns (x0, x1) of the first run wide enough to be a
    card back, or None.
    """
    colsum = blue_zone.sum(axis=0)  # number of blue pixels per column
    runs = []
    start = None
    for x in range(W):
        if colsum[x] >= 5:
            if start is None:
                start = x
        else:
            if start is not None:
                runs.append((start, x - 1))
                start = None
    if start is not None:
        runs.append((start, W - 1))
    for (x0, x1) in runs:
        if (x1 - x0 + 1) >= min_width:
            return (x0, x1)
    return None


class Layout:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.H, self.W = image.shape[:2]

        self.centers = find_tableau_columns(image)
        self.step = int(np.mean(np.diff(self.centers)))

        # measured card dimensions (resolution-adaptive)
        self.card_w, self.card_h = measure_card_size(image, self.centers)

        self.bottom_y = int(self.H * 0.90)  # stacks can reach far down (y~960 of 1040)

        # table_top_y ИЗМЕРЯЕТСЯ, а не угадывается как H*0.33: в Клондайке все
        # колонки начинаются на одной и той же высоте, поэтому медиана верхов
        # непустых стопок = точный верх стола. Прежняя оценка 0.33*H давала
        # 343 вместо реальных 350 и уводила дроп в пустую колонку НА 7px ВЫШЕ
        # пустого слота. Скан начинаем ниже верхнего ряда (stock/waste/
        # фундаменты заканчиваются ~0.29*H), чтобы не поймать их как «стопку».
        tops = [
            t for t in (
                find_stack_top(image, cx, int(self.H * 0.30), self.bottom_y, self.card_w)
                for cx in self.centers
            ) if t is not None
        ]
        self.table_top_y = int(np.median(tops)) if tops else int(self.H * 0.33)

        # y of each column's face-up card + number of face-down cards
        self.open_y = []
        self.closed = []
        for cx in self.centers:
            c, oy, _n = count_cards_in_column(
                image, cx, self.table_top_y, self.bottom_y, self.card_w, self.card_h
            )
            self.closed.append(c)
            self.open_y.append(oy)

        # top row: stock (blue back) + waste (fan of face-up cards) + foundations
        self._detect_top_row(image)

    def _detect_top_row(self, image):
        """
        Detect stock and waste positions in the top row (above the tableau).

        Stock = the blue card back on the far left of the top row.
        Waste = a fan of face-up cards to the right of stock (draw-3 mode).
        The card you can actually move is the RIGHTMOST (topmost) card of the
        fan — so waste_xy must point at its center, not at the middle.

        The top-row band is derived from the window height, so it adapts to
        any resolution (top row sits ~11%..29% down the window height).
        """
        r = image[:, :, 0].astype(int)
        g = image[:, :, 1].astype(int)
        b = image[:, :, 2].astype(int)

        top_band_top = int(self.H * 0.115)
        top_band_bottom = int(self.H * 0.29)
        top_zone = slice(top_band_top, top_band_bottom)

        # stock: blue back (b >> r and b >> g) in the top-left area.
        # Use the FIRST solid blue x-run (the far-left blue card), not the
        # average of every blue pixel (which drifts right as blue UI/backs
        # elsewhere get included), so the click always lands on the draw pile.
        blue = (b > r + 15) & (b > g)
        blue_zone = blue[top_zone, :]
        stock_run = _largest_blue_run(
            blue_zone, W=self.W, min_width=max(40, self.card_w // 2)
        )
        if stock_run is not None:
            x0, x1 = stock_run
            sub = blue_zone[:, x0:x1 + 1]
            ys, _ = np.where(sub)
            y0 = top_band_top + int(ys.min())
            y1 = top_band_top + int(ys.max())
            self.stock_xy = ((x0 + x1) // 2, (y0 + y1) // 2)
        else:
            self.stock_xy = (self.centers[0], top_band_top + self.card_h // 2)

        # waste: face-up white cards to the right of stock, but STRICTLY
        # LEFT of the first foundation card's LEFT EDGE (not its center).
        # Foundations render white when occupied, so only the zone strictly
        # left of `centers[3] - card_w` is the actual waste fan. Using the
        # center as the boundary still lets the foundation card's left half
        # (which is white) leak into the "waste" strip.
        found_left_x = (self.centers[3] - self.card_w) if len(self.centers) >= 7 else int(self.W * 0.40)
        white = (r > 200) & (g > 200) & (b > 200)
        white_zone = white[top_zone, :found_left_x]
        colsum = white_zone.sum(axis=0)
        clusters = []
        start = None
        for x in range(found_left_x):
            if colsum[x] > 10 and start is None:
                start = x
            elif colsum[x] <= 10 and start is not None:
                if x - start >= 40:
                    clusters.append((start, x - 1))
                start = None
        if start is not None and found_left_x - start >= 40:
            clusters.append((start, found_left_x - 1))

        if clusters:
            # waste = rightmost cluster that is still left of foundations.
            # The waste is a FAN: only the RIGHTMOST (topmost) card is fully
            # visible and playable; the cards to its left only show a sliver.
            # So the grab/recognize anchor must be the TOP card's center =
            # (right edge of the whole fan) - card_w/2, NOT the fan's x-center
            # (which points at a BURIED middle card — this caused the memory to
            # read 10♦ while the real top was 6♠).
            ls, le = clusters[-1]
            waste_cx = le - self.card_w // 2
            sub = white_zone[:, ls:le + 1]
            ys = np.where(sub.any(axis=1))[0]
            waste_cy = top_band_top + int(np.mean(ys)) if len(ys) else top_band_top + self.card_h // 2
            self.waste_xy = (int(waste_cx), int(waste_cy))
            # ТОЧНЫЙ верх верхней waste-карты (тот же find_card_top_y, что и в
            # screen_to_board) — чтобы кроп для распознавания начинался ровно
            # с края карты, а не на ~13px ниже (среднее по вееру).
            wtop = find_card_top_y(image, int(waste_cx), top_band_top, top_band_bottom)
            self.waste_top_y = int(wtop) if wtop is not None else top_band_top
            self.waste_left_x = ls
            self.waste_right_x = le
        else:
            self.waste_xy = (self.centers[0] + self.step, top_band_top + self.card_h // 2)
            self.waste_top_y = top_band_top

        # foundations: 4 slots in the top row, aligned with the RIGHTMOST
        # 4 tableau columns (centers[3..6]), matching screen_to_board. The
        # foundation slot top is the same as the top-band top (card tops are
        # flush with the top row). Slot order: ♥ ♦ ♣ ♠ (left -> right).
        found_top = top_band_top
        self.foundation_xy = [
            (self.centers[3 + i], found_top) for i in range(4)
        ] if len(self.centers) >= 7 else [
            (self.centers[0] + self.step * (2 + i), found_top) for i in range(4)
        ]

    def tableau_card_xy(self, col: int) -> tuple[int, int]:
        """
        Coordinates of the CENTER of a column's top face-up card (col 0..6).

        Grabs the top face-up card by its center = open_y + card_h/2.
        For a stack of several face-up cards, grabbing the top card also
        drags the whole face-up run below it (the game picks it up).
        """
        cx = self.centers[col]
        oy = self.open_y[col]
        if oy is None:
            # пустая колонка — это ЦЕЛЬ для короля. Возвращаем ЦЕНТР пустого
            # слота: дроп у самого его верхнего края (table_top_y) ненадёжен —
            # курсор оказывается на кромке/выше слота и игра бросок не принимает.
            return cx, self.table_top_y + self.card_h // 2
        y = oy + self.card_h // 2
        return cx, y

    def foundation_xy_for(self, suit_rank_index: int) -> tuple[int, int]:
        """Coordinates of the n-th foundation pile (0..3)."""
        return self.foundation_xy[suit_rank_index]
