"""
Bridge "screenshot → Board".

Takes an RGB screenshot of the table and builds a rule_engine Board:
- 7 tableau columns: how many cards in each (face-down backs + top face-up);
- the face-up card is recognized via the trained MobileNet;
- stock, waste and foundations — by position.

Face-down cards are represented as Card(UNKNOWN_RANK, Suit.UNKNOWN, face_up=False)
— by Klondike rules their rank/suit are not needed until the card is flipped.
"""

from __future__ import annotations

import numpy as np

from orchestrator.rule_engine import (
    REAL_SUITS,
    UNKNOWN_RANK,
    Board,
    Card,
    Suit,
)


def _is_green(r, g, b):
    """Works with both scalars and numpy arrays."""
    return (g > r + 20) & (g > b + 20)


def _is_white(r, g, b):
    return (r > 200) & (g > 200) & (b > 200)


def _is_blue(r, g, b):
    return (b > r + 15) & (b > g)


def measure_card_size(image: np.ndarray, centers: list[int]) -> tuple[int, int]:
    """
    Measure the real width and height of a face-up card from the screenshot.

    Column spacing (distance between adjacent column centers) is a reliable
    anchor: in every Microsoft Solitaire layout the 7 columns are evenly
    spaced, so `step = mean(diff(centers))`. The card is a fixed fraction of
    that spacing regardless of screen resolution:
        card_w ≈ step * 0.78   (measured: 132 / 168)
        card_h ≈ step * 1.05   (measured: 176 / 168)

    Returns (card_w, card_h) in pixels for THIS screenshot.
    """
    if len(centers) < 2:
        return 131, 176
    step = int(round(np.mean(np.diff(centers))))
    card_w = max(20, int(round(step * 0.78)))
    card_h = max(20, int(round(step * 1.05)))
    return card_w, card_h


def find_tableau_columns(image: np.ndarray, expected: int = 7) -> list[int]:
    """
    Find the x-centers of the 7 tableau columns.

    First try to detect `expected` wide clusters on a single row (works for a
    full board). If that fails (some columns empty), reconstruct the grid from
    the columns we DID find: the columns are evenly spaced, so we extend the
    measured spacing to the left/right to fill in missing (empty) columns.
    This is resolution-independent — it only uses the measured spacing, not
    hardcoded screen-ratio constants.
    """
    H, W = image.shape[:2]
    r = image[:, :, 0].astype(int)
    g = image[:, :, 1].astype(int)
    b = image[:, :, 2].astype(int)
    not_green = ~_is_green(r, g, b)
    min_cluster = W // (expected * 5)

    # honest scan: a row where exactly `expected` wide clusters appear
    for y in range(int(H * 0.35), int(H * 0.75), 4):
        row_mask = not_green[y]
        clusters = []
        cur = []
        for x in range(W):
            if row_mask[x]:
                cur.append(x)
            elif cur:
                clusters.append(cur)
                cur = []
        if cur:
            clusters.append(cur)
        centers = sorted(int(np.mean(c)) for c in clusters if len(c) >= min_cluster)
        if len(centers) == expected:
            return centers

    # fallback: we found SOME columns (or none). Reconstruct using the REAL
    # spacing between the detected clusters. This is robust to empty columns
    # and to any screen resolution, because it uses the measured card geometry.
    best_centers: list[int] = []
    best_widths: list[int] = []
    for y in range(int(H * 0.35), int(H * 0.75), 4):
        row_mask = not_green[y]
        clusters = []
        cur = []
        for x in range(W):
            if row_mask[x]:
                cur.append(x)
            elif cur:
                clusters.append(cur)
                cur = []
        if cur:
            clusters.append(cur)
        wide = [c for c in clusters if len(c) >= min_cluster]
        centers = sorted(int(np.mean(c)) for c in wide)
        if len(centers) > len(best_centers):
            best_centers = centers
            # ширина клубка нужна как независимая оценка шага сетки
            best_widths = sorted(c[-1] - c[0] + 1 for c in wide)

    if len(best_centers) == 0:
        # no columns visible at all — nothing to anchor on
        raise ValueError("Could not find any tableau columns to calibrate.")

    if len(best_centers) == 1:
        # only one column: assume a spacing of ~W/8 (7 cols across the board)
        step = max(20, int(W * 0.0875))
        centers = [best_centers[0] + step * (i - 3) for i in range(expected)]
        return [c for c in centers if 20 <= c < W - 20]

    # ── шаг сетки ────────────────────────────────────────────────────────
    # КЛЮЧЕВОЕ: пустая колонка НЕ исчезает из сетки — она просто не даёт
    # клубка. Поэтому зазоры между найденными колонками кратны шагу, и шаг
    # равен МИНИМАЛЬНОМУ зазору. Прежний `median(gaps)` на пустом столе давал
    # полтора шага (зазоры [1,2,2,1] шага -> медиана 1.5 шага), сетка уезжала
    # (centers 203,455,707,959,1211,1463,1715 вместо 455..1463 с шагом 168) и
    # ВСЁ чтение стола превращалось в мусор сразу после первой пустой колонки.
    gaps = np.diff(best_centers)
    step = int(round(np.min(gaps)))

    # независимая проверка шага: ширина карты ≈ 0.78 шага (измерено на живой
    # игре: 131px карта при шаге 168px). Если минимум зазоров противоречит
    # ширине (например, видны всего две далёкие колонки) — верим ширине.
    if best_widths:
        step_from_width = int(round(float(np.median(best_widths)) / 0.78))
        if step_from_width >= 20 and not (0.8 * step_from_width <= step <= 1.25 * step_from_width):
            step = step_from_width
    if step < 20:
        step = max(20, int(W // 8))

    window_center = W / 2
    tol = max(6, step // 6)
    best = None
    best_score = None
    # try: the leftmost detected column IS some col index k (0..expected-1).
    # Сетка выбирается по ЧИСЛУ СОВПАВШИХ колонок (а не только по центровке —
    # при неверном шаге центровка тоже нулевая и мусорная сетка выигрывала).
    for k in range(expected):
        col0 = best_centers[0] - k * step
        grid = [col0 + i * step for i in range(expected)]
        if grid[0] < step // 2 or grid[-1] > W - step // 2:
            continue
        matches = sum(1 for c in best_centers if min(abs(c - gi) for gi in grid) <= tol)
        grid_center = (grid[0] + grid[-1]) / 2
        score = (matches, -abs(grid_center - window_center))
        if best_score is None or score > best_score:
            best_score = score
            best = grid

    # Если ни одна сетка не влезла в кадр, значит стол не виден корректно —
    # не возвращаем мусорные центры (они дают cx>W и роняют
    # count_cards_in_column), а честно сигналим об ошибке.
    if best is None:
        raise ValueError("Table grid does not fit the window — table not visible.")

    return [int(round(c)) for c in best]


def find_card_top_y(image: np.ndarray, cx: int, y_from: int = 100, y_to: int = 300) -> int | None:
    """
    Find the y where a card starts in the top zone (stock/waste/foundations).

    Scans a small horizontal WINDOW around cx (not a single pixel column), and
    returns the TOPMOST white pixel in that window. A single-column scan is
    fragile: a large rank symbol (K/Q/J) sits right in the middle of the card,
    so the white background is interrupted there and a naive "first >=12-px
    white run" returns the symbol's lower edge instead of the card's true top.
    """
    H = image.shape[0]
    W = image.shape[1]
    x0 = max(0, cx - 25)
    x1 = min(W, cx + 26)
    region = image[y_from:min(y_to, H), x0:x1, :3].astype(int)
    white = (region[:, :, 0] > 200) & (region[:, :, 1] > 200) & (region[:, :, 2] > 200)
    ys, _ = np.where(white)
    if len(ys) == 0:
        return None
    return int(y_from + ys.min())


def find_stack_bottom(
    image: np.ndarray, cx: int | None, top_y: int, bottom_y: int, card_w: int = 131
) -> int | None:
    """
    y of the BOTTOM edge of the column's card stack (None if the column is empty).

    A column is a single vertical stack of overlapping cards sitting on the
    felt, so "the stack" is simply every non-green row in a horizontal window
    around the column center. The BOTTOM card of a Klondike column is the one
    card that is ALWAYS fully visible (nothing overlaps it from below), so this
    edge is the one anchor that does not depend on how many cards the fan has.

    Returns the largest such y, or None when the window is pure felt (empty
    column). A 1-2px decorative outline of an empty slot cannot pass the 0.5
    fill threshold, so empty columns stay empty.
    """
    H, W = image.shape[:2]
    if cx is None or not (0 <= cx < W):
        return None
    y0 = max(0, int(top_y))
    y1 = min(H, int(bottom_y))
    if y1 <= y0:
        return None
    half = max(10, card_w // 3)
    x0 = max(0, cx - half)
    x1 = min(W, cx + half)
    region = image[y0:y1, x0:x1, :3].astype(int)
    rr = region[:, :, 0]
    gg = region[:, :, 1]
    bb = region[:, :, 2]
    is_green = (gg > rr + 20) & (gg > bb + 20)
    not_green_frac = 1.0 - is_green.mean(axis=1)
    rows = np.where(not_green_frac >= 0.5)[0]
    if len(rows) == 0:
        return None
    return int(y0 + rows.max())


def find_stack_top(
    image: np.ndarray, cx: int | None, top_y: int, bottom_y: int, card_w: int = 131
) -> int | None:
    """
    y of the TOP edge of the column's card stack (None if the column is empty).

    Symmetric to `find_stack_bottom`. In Klondike every column starts at the
    SAME y (the tableau top), so the median of these tops over the non-empty
    columns is an exact measurement of `table_top_y` — unlike the old
    `int(H * 0.33)` guess, which was 7px off (343 vs the real 350) and made
    "drop onto an empty column" land ABOVE the empty slot.
    """
    H, W = image.shape[:2]
    if cx is None or not (0 <= cx < W):
        return None
    y0 = max(0, int(top_y))
    y1 = min(H, int(bottom_y))
    if y1 <= y0:
        return None
    half = max(10, card_w // 3)
    x0 = max(0, cx - half)
    x1 = min(W, cx + half)
    region = image[y0:y1, x0:x1, :3].astype(int)
    rr = region[:, :, 0]
    gg = region[:, :, 1]
    bb = region[:, :, 2]
    is_green = (gg > rr + 20) & (gg > bb + 20)
    not_green_frac = 1.0 - is_green.mean(axis=1)
    rows = np.where(not_green_frac >= 0.5)[0]
    if len(rows) == 0:
        return None
    return int(y0 + rows.min())


def _card_is_back(image: np.ndarray, cx: int, top_y: int, card_w: int, card_h: int) -> bool:
    """
    Is the card whose top is `top_y` a face-down (blue) back?

    Признак считается по ВСЕЙ площади карты (с отступом от края): рубашка синяя
    от края до края (измерено на живой игре: 1.000), а открытая карта — белый
    фон с рисунком. Узкая полоса по центру здесь НЕ годится: у фигур (Q♣, J♦,
    K♥...) портрет нарисован в синем/фиолетовом и даёт >0.5 синего в центре,
    из-за чего Q♣ принималась за рубашку и вся колонка читалась как «закрыта»
    (измерено: Q♣ = 0.214 по всей карте против 1.000 у рубашки).
    """
    H, W = image.shape[:2]
    inset = 8
    x0 = max(0, cx - card_w // 2 + inset)
    x1 = min(W, cx + card_w // 2 - inset)
    y0 = max(0, top_y + inset)
    y1 = min(H, top_y + card_h - inset)
    if y1 <= y0 or x1 <= x0:
        return False
    region = image[y0:y1, x0:x1, :3].astype(int)
    rr = region[:, :, 0]
    gg = region[:, :, 1]
    bb = region[:, :, 2]
    is_blue = (bb > rr + 15) & (bb > gg)
    return bool(is_blue.mean() > 0.5)


def count_cards_in_column(image: np.ndarray, cx: int, top_y: int, bottom_y: int, card_w: int = 131, card_h: int = 176):
    """
    Count cards in a column.

    Returns (num_face_down, y_of_playable_card_top, n_open).

    ⚠️ CONTRACT: `open_y` is the top of the BOTTOM-most (= PLAYABLE, "free")
    face-up card of the column — NOT the top of the first face-up card under
    the backs. `layout.open_y`, `verify.check_shift`, `verify_quantity`,
    `apply_move_to_screen` and `screen_to_board` all treat it as the free card,
    so returning the fan's first card instead desynchronised memory, clicks and
    verification at once (the "mem=X screen=Y -> Stuck" loop).

    The free card is found geometrically, without counting the fan: the bottom
    card of a column is the only fully-visible card there, so its top is
    `stack_bottom - card_h + 1`. That works for any fan size (1 card or 7) and
    does not depend on the model recognising overlapped cards.

    A face-down card is a BLUE back. The count of backs is derived from the
    structure "backs are a solid group on TOP, the first face-up card starts at
    the first long white run" — see the comments below.
    """
    W = image.shape[1]
    # Защита от мусорного cx (вне кадра): find_tableau_columns может вернуть
    # центр за пределами [0, W), когда стол не виден (окно свёрнуто/белый фон).
    # Без этой проверки image[:, cx] кидает IndexError и роняет весь /move.
    if cx is None or not (0 <= cx < W):
        return 0, None, 0

    # Count face-down cards by structure, NOT by blue-run length.
    #
    # В колонке рубашки идут СПЛОШНОЙ группой СВЕРХУ, а первая открытая карта
    # начинается на ДЛИННОМ БЕЛОМ участке (белый фон карты). Символы масти на
    # открытой карте (♠/♣ — тёмно-синие) дают синие прогоны ЛЮБОЙ длины
    # (1..15px), неотличимые от тонких рубашек — поэтому "рубашка = синий
    # прогон длиннее порога" НЕ работает (12px символ A♠ ложно был принят за
    # рубашку, bug col4: "4 closed" вместо 3, open_y улетал на символ масти).
    #
    # Надёжный признак — ПОРЯДОК, а не длина: рубашки идут ДО первой открытой
    # карты. Ищем первый длинный белый прогон (начало открытой карты) и считаем
    # рубашками только синие прогоны ВЫШЕ него.
    c = image[top_y:bottom_y, cx, 2].astype(int)
    cr = image[top_y:bottom_y, cx, 0].astype(int)
    cg = image[top_y:bottom_y, cx, 1].astype(int)
    blue_line = (c > cr + 15) & (c > cg)

    # белый = яркий фон открытой карты. Длинный белый прогон = начало открытой
    # карты (белый фон между рубашками и символами длиннее, чем белые просветы
    # ВНУТРИ веера рубашек).
    rr_line = image[top_y:bottom_y, cx, 0].astype(int)
    white_line = rr_line > 200
    min_white_run = max(8, int(round(card_h * 0.06)))  # ~10px при card_h=176

    # первый длинный белый прогон = граница "рубашки кончились, открытая началась"
    first_open_start = None
    ws = None
    for i in range(len(white_line)):
        if white_line[i]:
            if ws is None:
                ws = i
        else:
            if ws is not None:
                if i - ws >= min_white_run:
                    first_open_start = ws
                    break
                ws = None
    if ws is not None and first_open_start is None:
        if len(white_line) - ws >= min_white_run:
            first_open_start = ws

    if first_open_start is None:
        # нет открытой карты вообще (пустая колонка или вся колонка закрыта).
        # Считаем все длинные синие прогоны рубашками (открытых карт нет).
        first_open_start = len(blue_line)

    # считаем рубашки ТОЛЬКО выше первой открытой карты, по длинным прогонам
    min_back_run = max(8, int(round(card_h * 0.07)))  # ~12px при card_h=176
    closed = 0
    run_start = None
    for i in range(first_open_start):
        if blue_line[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if i - run_start >= min_back_run:
                    closed += 1
                run_start = None
    if run_start is not None:
        if first_open_start - run_start >= min_back_run:
            closed += 1

    # ИГРАБЕЛЬНАЯ (переносимая) КАРТА = НИЖНЯЯ карта колонки. Она всегда
    # полностью видима (её ничто не перекрывает снизу), поэтому её верх
    # считается чистой геометрией: низ стопки минус высота карты. Это не
    # зависит от размера веера открытых карт — в отличие от «первой открытой
    # карты под рубашками», которая при веере ≥2 давала НЕ играбельную карту
    # (и рассыпала клики, verify и память одновременно).
    stack_bottom = find_stack_bottom(image, cx, top_y, bottom_y, card_w)
    if stack_bottom is None:
        return 0, None, 0  # колонка пуста — сукно

    playable_top = int(stack_bottom - card_h + 1)

    # Если нижняя карта — рубашка вверх, открытых карт в колонке нет (в
    # Клондайке так бывает только в короткое окно анимации, но обрабатываем
    # честно, чтобы не «распознать» рубашку как ранг).
    if _card_is_back(image, cx, playable_top, card_w, card_h):
        return closed, None, 0

    return closed, playable_top, 1


def build_board_from_columns(
    image: np.ndarray,
    centers: list[int],
    table_top_y: int | None = None,
    recognize: callable = None,
) -> Board:
    """
    Build a Board from 7 columns.

    recognize — callable(image, cx, top_y, card_w, card_h) -> (rank, suit)
    for the face-up card (usually recognizer.recognize_at). Face-down cards
    are placeholders. All geometry is derived from the measured card size, so
    this adapts to any screen resolution.
    """
    H, W = image.shape[:2]
    card_w, card_h = measure_card_size(image, centers)
    if table_top_y is None:
        table_top_y = int(H * 0.33)  # top of the stacks
    bottom_y = int(H * 0.90)  # stacks can reach down to ~y=960; scan that deep

    board = Board()

    # 1) fill the 7 tableau columns. Only the TOP (playable = bottommost) open
    # card of each column is needed for move generation. count_cards_in_column
    # returns that card's top y directly; we crop and recognize it in one shot.
    for col_idx, cx in enumerate(centers):
        closed, first_open, n_open = count_cards_in_column(
            image, cx, table_top_y, bottom_y, card_w, card_h
        )
        col = []
        for _ in range(closed):
            col.append(Card(UNKNOWN_RANK, Suit.UNKNOWN, face_up=False))
        if first_open is not None:
            try:
                res = recognize(image, cx, first_open, card_w, card_h)
                rank, suit = res[0], res[1]
                top = Card(rank, suit, face_up=True)
            except Exception:
                top = Card(UNKNOWN_RANK, Suit.UNKNOWN, face_up=True)
            col.append(top)
        board.tableau[col_idx] = col

    # 2) stock — blue card back on the far left of the top row. Position is
    # also where the "recycle" (green circle) appears once the stock is empty.
    # Detect the stock/recycle control by its x (same as the leftmost column).
    stock_x = centers[0]
    top_band_top = int(H * 0.115)  # ~120px at 1040 -> top of the top row
    top_band_bottom = int(H * 0.29)  # ~300px
    r = int(image[top_band_top + card_h // 2, stock_x, 0])
    g = int(image[top_band_top + card_h // 2, stock_x, 1])
    b = int(image[top_band_top + card_h // 2, stock_x, 2])
    has_stock = not (g > r + 20 and g > b + 20)  # not felt = a card/back is there
    if has_stock:
        # exact number in stock doesn't matter for moves — only that it exists
        board.stock = [Card(UNKNOWN_RANK, Suit.UNKNOWN, face_up=False) for _ in range(24)]

    # 3) waste — the white card cluster RIGHT of stock but STRICTLY LEFT of
    # the first foundation card's LEFT EDGE. Foundations render white when
    # occupied; using `centers[3]` (the center) as the boundary still lets the
    # foundation card's white left half leak in once the waste fan is empty.
    step = int(np.mean(np.diff(centers))) if len(centers) > 1 else 0
    found_left_x = (centers[3] - card_w) if len(centers) >= 7 else W // 2
    white = _is_white(image[:, :, 0].astype(int), image[:, :, 1].astype(int), image[:, :, 2].astype(int))
    white_top = white[top_band_top:top_band_bottom, :found_left_x]
    colsum = white_top.sum(axis=0)
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
        # The waste is a FAN — the playable card is the RIGHTMOST (topmost),
        # fully-visible one; cards to its left are only slivers. Recognize at
        # the top card's center = right fan edge - card_w/2 (NOT the fan center,
        # which lands on a buried card).
        ls, le = clusters[-1]  # rightmost cluster left of foundations = waste
        width = le - ls + 1
        # How many cards are in the fan? Each extra card shifts the fan and
        # widens the cluster by ~0.103 * card_w (≈13px @ card_w=131). This lets
        # us recover the COUNT (1..3) from geometry alone, so memory knows the
        # real fan size even when re-reading mid-game (a plain 1-card read lost
        # the 5♠/Q♣ below the top 4♣).
        waste_step = max(6, round(card_w * 0.103))
        n_waste = round((width - card_w) / waste_step) + 1
        n_waste = max(1, min(3, n_waste))

        waste_x = le - card_w // 2
        ctop = find_card_top_y(image, waste_x, top_band_top, top_band_bottom)
        top_card = None
        if ctop is not None and recognize is not None:
            res = recognize(image, waste_x, ctop, card_w, card_h)
            rank, suit = res[0], res[1]
            top_card = Card(rank, suit, face_up=True)
        if top_card is not None:
            # fan: (n-1) buried UNKNOWN cards below the identified top card
            board.waste = [Card(UNKNOWN_RANK, Suit.UNKNOWN, face_up=True)
                           for _ in range(n_waste - 1)] + [top_card]

    # 4) foundations — 4 slots in the top row, aligned with the RIGHTMOST
    # 4 tableau columns (centers[3..6]). IMPORTANT: slots are NOT bound to a
    # fixed suit — the suit is set by the first card (an ace) placed there,
    # in whatever order the player plays them. So we read the card in each
    # slot and key the foundation dict by the card's ACTUAL suit, not by the
    # slot index. (Previously slots were hard-wired to ♥♦♣♠, which mis-mapped
    # a second ace and made the engine "move" it back down.)
    for i in range(4):
        fx = centers[3 + i] if len(centers) >= 7 else centers[0] + step * (2 + i)
        if fx >= W:
            continue
        ctop = find_card_top_y(image, fx, top_band_top, top_band_bottom)
        if ctop is not None and recognize is not None:
            res = recognize(image, fx, ctop, card_w, card_h)
            rank, r_suit = res[0], res[1]
            if r_suit in REAL_SUITS:
                board.foundation[r_suit] = [Card(rank, r_suit, face_up=True)]

    return board


def screen_to_board(image: np.ndarray, recognizer=None) -> Board:
    """
    Full bridge: RGB screenshot of the table -> Board.

    Finds the 7 columns itself and assembles the state. recognizer is a
    Recognizer object (if None, it loads one).
    """
    if recognizer is None:
        from vision.recognizer import get_recognizer
        recognizer = get_recognizer()

    centers = find_tableau_columns(image)
    return build_board_from_columns(
        image, centers, recognize=recognizer.recognize_at_with_confidence
    )
