"""
Регресс-тест vision-слоя на РЕАЛЬНОМ кадре игры (не синтетика).

Кадр `tests/data/live_frame_2026-10-01.png` снят с живой партии Microsoft
Solitaire Collection (1920x1040) и вручную прочитан глазами. Это эталон:

    [1] 9♠ 8♦ 7♠ 6♥             0 рубашек, играбельная = 6♥
    [2] ? 8♠                    1 рубашка,  играбельная = 8♠
    [3] ? 4♠                    1 рубашка,  играбельная = 4♠
    [4] ? ? ? K♦ Q♠ J♦ 10♣     3 рубашки,  играбельная = 10♣
    [5] ? ? ? Q♦                3 рубашки,  играбельная = Q♦
    [6] ? ? 3♦                  2 рубашки,  играбельная = 3♦
    [7] ? ? ? ? ? ? 10♠         6 рубашек,  играбельная = 10♠
    waste = 4♣ K♥ 9♥ (верх 9♥), foundations ♥:A♥ ♦:A♦

Смысл теста: `count_cards_in_column` ОБЯЗАН возвращать верх НИЖНЕЙ
(играбельной) карты колонки, а не первой открытой карты веера. Именно эта
ошибка ("open_y = первая открытая") рассыпала клики, verify и память и
приводила к вечному «mem=X screen=Y → Stuck».
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
from PIL import Image

from orchestrator.rule_engine import RANK_NAMES, Suit
from vision import screen_to_board as s2b

FRAME = Path(__file__).resolve().parent / "data" / "live_frame_2026-10-01.png"
# Второй кадр — та же игра позже, когда колонки 2 и 6 УЖЕ ПУСТЫ. На нём
# ломался find_tableau_columns: median(gaps) по [1,2,2,1] шага давал 1.5 шага,
# сетка уезжала (203,455,707,959,1211,1463,1715) и всё чтение стола становилось
# мусором сразу после первой пустой колонки.
FRAME_EMPTY = Path(__file__).resolve().parent / "data" / "live_frame_empty_cols_2026-10-01.png"
# Третий кадр — СВЕЖАЯ раздача (счёт 0:00). На ней ловится регресс «синего
# портрета»: нижняя карта колонки 5 — Q♣, у которой платье нарисовано синим.
# Проверка «рубашка» по центральной полосе давала >0.5 синего и вся колонка
# читалась как закрытая (4 рубашки и НЕТ открытой карты).
FRAME_FRESH = Path(__file__).resolve().parent / "data" / "live_frame_fresh_deal_2026-10-01.png"

# (колонка 1-based, число рубашек, карта-верх, нижняя граница стопки)
GROUND_TRUTH = [
    (1, 0, "6♥", 689),
    (2, 1, "8♠", 543),
    (3, 1, "4♠", 543),
    (4, 3, "10♣", 742),
    (5, 3, "Q♦", 578),
    (6, 2, "3♦", 560),
    (7, 6, "10♠", 631),
]


def _load():
    image = np.array(Image.open(FRAME).convert("RGB"))
    centers = s2b.find_tableau_columns(image)
    card_w, card_h = s2b.measure_card_size(image, centers)
    return image, centers, card_w, card_h


def test_geometry_matches_ground_truth():
    """Геометрия колонок и низ стопки — без ML, чистые пиксели."""
    image, centers, card_w, card_h = _load()
    assert len(centers) == 7, f"должно быть 7 колонок, найдено {len(centers)}"
    assert (card_w, card_h) == (131, 176), f"размер карты {card_w}x{card_h}"

    H = image.shape[0]
    top_y, bottom_y = int(H * 0.33), int(H * 0.90)
    for idx, cx in enumerate(centers):
        col_no, exp_closed, _name, exp_bottom = GROUND_TRUTH[idx]
        stack_bottom = s2b.find_stack_bottom(image, cx, top_y, bottom_y, card_w)
        assert stack_bottom == exp_bottom, (
            f"колонка {col_no}: низ стопки {stack_bottom}, ожидался {exp_bottom}")

        closed, open_y, _ = s2b.count_cards_in_column(
            image, cx, top_y, bottom_y, card_w, card_h)
        assert closed == exp_closed, (
            f"колонка {col_no}: рубашек {closed}, ожидалось {exp_closed}")
        assert open_y == exp_bottom - card_h + 1, (
            f"колонка {col_no}: open_y={open_y} — это НЕ верх нижней карты "
            f"(ожидался {exp_bottom - card_h + 1})")


def test_playable_card_is_recognized_on_every_column():
    """Играбельная (нижняя) карта каждой колонки распознаётся верно.

    Это и есть регресс: до фикса модель получала на вход ПЕРЕКРЫТУЮ карту
    (верх веера) и путала масть (K♦ → K♠, conf 0.254), а память сервера
    расходилась с экраном → «Stuck».
    """
    from vision.recognizer import get_recognizer

    image, centers, card_w, card_h = _load()
    rec = get_recognizer()
    H = image.shape[0]
    top_y, bottom_y = int(H * 0.33), int(H * 0.90)

    for idx, cx in enumerate(centers):
        col_no, _closed, exp_name, _bottom = GROUND_TRUTH[idx]
        _c, open_y, _n = s2b.count_cards_in_column(
            image, cx, top_y, bottom_y, card_w, card_h)
        rank, suit, conf = rec.recognize_at_with_confidence(
            image, cx, open_y, card_w, card_h)
        got = f"{RANK_NAMES.get(rank, rank)}{suit.value}"
        assert got == exp_name, (
            f"колонка {col_no}: распознано {got}, ожидалось {exp_name} (conf={conf:.3f})")
        assert conf > 0.9, (
            f"колонка {col_no}: уверенность {conf:.3f} слишком низкая — "
            "похоже, в кроп попала перекрытая карта")


def test_full_board_read_is_consistent():
    """Полное чтение стола: играбельные карты колонок + waste + фундаменты."""
    from vision.recognizer import get_recognizer

    image, _centers, _w, _h = _load()
    board = s2b.screen_to_board(image, recognizer=get_recognizer())

    tops = []
    for col in board.tableau:
        open_cards = [c for c in col if c.face_up]
        tops.append(open_cards[-1] if open_cards else None)
    got = [f"{RANK_NAMES.get(c.rank, c.rank)}{c.suit.value}" if c else None for c in tops]
    assert got == ["6♥", "8♠", "4♠", "10♣", "Q♦", "3♦", "10♠"], got

    assert board.waste and board.waste[-1].rank == 9 and board.waste[-1].suit == Suit.HEARTS

    assert board.foundation[Suit.HEARTS][-1].rank == 1
    assert board.foundation[Suit.DIAMONDS][-1].rank == 1
    assert board.foundation[Suit.CLUBS] == []
    assert board.foundation[Suit.SPADES] == []

    # закрытых карт ровно столько, сколько рубашек на экране
    closed = [sum(1 for c in col if not c.face_up) for col in board.tableau]
    assert closed == [0, 1, 1, 3, 3, 2, 6], closed


def test_empty_column_is_empty():
    """Пустая колонка (чистое сукно) не должна «находить» карту."""
    image, centers, card_w, card_h = _load()
    H, W = image.shape[:2]
    # сдвигаем окно в чистое сукно между игровым полем и нижней панелью
    assert s2b.find_stack_bottom(image, centers[0], int(H * 0.86), int(H * 0.90),
                                 card_w) is None
    # и координата вне кадра не роняет функцию
    assert s2b.find_stack_bottom(image, W + 50, 300, 900, card_w) is None
    assert s2b.count_cards_in_column(image, W + 50, 300, 900, card_w, card_h) == (0, None, 0)


# ── кадр с ПУСТЫМИ колонками ────────────────────────────────────────────
# Эталон (глазами по live_frame_empty_cols_2026-10-01.png):
#   stock пуст (зелёный круг), waste = 4♦ 9♣ J♣ (верх J♣)
#   foundations ♥:A♥ ♦:A♦ ♣:A♣
#   [1] 9♠ 8♦ 7♠ 6♥ 5♥ 4♥ 3♥ 2♥ 9♥? -> верх 6♥ (проверено: open_y=514)
#   [2] ПУСТА   [3] ? 4♠ 3♦ 2♠   [4] ? ? ? K♦ Q♠ J♦ 10♣ 9♥ 8♠
#   [5] ? ? ? Q♦ J♠   [6] ПУСТА   [7] ? ? ? ? ? ? 10♠
EMPTY_TRUTH = [
    (1, 0, "6♥"), (2, None, None), (3, 1, "2♠"), (4, 3, "8♠"),
    (5, 3, "J♠"), (6, None, None), (7, 6, "10♠"),
]


def test_empty_columns_do_not_break_the_grid():
    """Пустые колонки НЕ должны сдвигать сетку и размер карты."""
    image = np.array(Image.open(FRAME_EMPTY).convert("RGB"))
    centers = s2b.find_tableau_columns(image)
    assert centers == [455, 623, 791, 959, 1127, 1295, 1463], centers
    assert s2b.measure_card_size(image, centers) == (131, 176)


def test_empty_columns_are_read_as_empty():
    """Пустая колонка = (0, None, 0), непустые — со своей играбельной картой."""
    from vision.recognizer import get_recognizer

    image = np.array(Image.open(FRAME_EMPTY).convert("RGB"))
    centers = s2b.find_tableau_columns(image)
    card_w, card_h = s2b.measure_card_size(image, centers)
    rec = get_recognizer()
    H = image.shape[0]
    top_y, bottom_y = int(H * 0.33), int(H * 0.90)

    for col_no, exp_closed, exp_name in EMPTY_TRUTH:
        i = col_no - 1
        closed, open_y, n_open = s2b.count_cards_in_column(
            image, centers[i], top_y, bottom_y, card_w, card_h)
        if exp_closed is None:
            assert (closed, open_y, n_open) == (0, None, 0), (
                f"колонка {col_no} должна быть ПУСТОЙ, получено "
                f"({closed}, {open_y}, {n_open})")
            continue
        assert closed == exp_closed, f"колонка {col_no}: рубашек {closed} != {exp_closed}"
        rank, suit, conf = rec.recognize_at_with_confidence(
            image, centers[i], open_y, card_w, card_h)
        got = f"{RANK_NAMES.get(rank, rank)}{suit.value}"
        assert got == exp_name and conf > 0.9, (
            f"колонка {col_no}: {got} (conf={conf:.3f}), ожидалось {exp_name}")


# ── кадр СВЕЖЕЙ раздачи ────────────────────────────────────────────────
# Эталон: stock полон, waste и фундаменты пусты, а колонки 1..7 = 0..6 рубашек
# + ровно одна открытая карта (это и есть определение свежей раздачи).
FRESH_TRUTH = ["5♥", "2♥", "A♣", "J♦", "Q♣", "10♥", "9♣"]


def test_fresh_deal_is_read_exactly():
    """Свежая раздача: 0..6 рубашек, все открытые карты верные, Q♣ не «рубашка»."""
    from vision.recognizer import get_recognizer

    image = np.array(Image.open(FRAME_FRESH).convert("RGB"))
    centers = s2b.find_tableau_columns(image)
    assert centers == [455, 623, 791, 959, 1127, 1295, 1463], centers
    card_w, card_h = s2b.measure_card_size(image, centers)
    rec = get_recognizer()

    board = s2b.screen_to_board(image, recognizer=rec)
    closed = [sum(1 for c in col if not c.face_up) for col in board.tableau]
    assert closed == [0, 1, 2, 3, 4, 5, 6], closed

    tops = []
    for col in board.tableau:
        opens = [c for c in col if c.face_up]
        tops.append(f"{RANK_NAMES.get(opens[-1].rank, opens[-1].rank)}{opens[-1].suit.value}"
                    if opens else None)
    assert tops == FRESH_TRUTH, tops

    # свежая раздача = пустые waste/фундаменты + полный stock
    assert board.waste == []
    assert all(pile == [] for pile in board.foundation.values())
    assert board.stock, "stock свежей раздачи не должен быть пустым"

    # в каждой колонке открытая карта ЕСТЬ и её верх = низ стопки - card_h + 1
    for cx in centers:
        c, oy, n = s2b.count_cards_in_column(image, cx, 350, 936, card_w, card_h)
        assert oy is not None and n == 1, f"cx={cx}: open_y={oy}, n_open={n}"


def test_layout_measures_table_top_and_empty_slot():
    """Layout: верх стола ИЗМЕРЯЕТСЯ (350, не 343) и дроп в пустую — в центр слота."""
    from act.layout import Layout

    image = np.array(Image.open(FRAME_EMPTY).convert("RGB"))
    layout = Layout(image)
    assert layout.table_top_y == 350, layout.table_top_y
    assert layout.open_y[1] is None and layout.open_y[5] is None
    # цель для короля — центр пустого слота, а не его верхняя кромка
    assert layout.tableau_card_xy(1) == (623, 350 + layout.card_h // 2)
    assert layout.tableau_card_xy(5) == (1295, 350 + layout.card_h // 2)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[OK]   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} тестов прошло")
    sys.exit(1 if failed else 0)
