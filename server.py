"""
Local web server for the Klondike AI (FINAL version).

Now wired to the real orchestrator: the "Make a move" button captures the
game screen, builds a Board, asks the LLM and MOVES the cards with the mouse.
Reasoning and board state are returned to the browser.

Usage:  python server.py   (or double-click start.bat)
Then open in a browser:  http://localhost:8000
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from act import controller as act
from act.layout import Layout
from orchestrator import verify as vfy
from orchestrator.loop_guard import LoopGuard
from orchestrator.quantity_guard import FanStepEstimator, verify_quantity
from orchestrator.rule_engine import apply_move, get_valid_moves, render
from planner.llm_planner import (
    LLMBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
    choose_move,
)
from vision.recognizer import get_recognizer
from vision.screen_to_board import screen_to_board


def read_env() -> dict:
    """Read key=value pairs from .env into a dict."""
    values = {}
    for line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip()
    return values


def _log_move(msg: str):
    """Append a diagnostic line to move_log.log (survives browser aborts)."""
    import datetime
    try:
        with open(_ROOT / "move_log.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def build_backend() -> LLMBackend:
    env = read_env()
    backend = env.get("LLM_BACKEND", "ollama").lower()

    if backend == "deepseek":
        key = env.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise SystemExit("DEEPSEEK_API_KEY not found in .env")
        return OpenAICompatibleBackend(
            model_name="deepseek-chat",
            api_key=key,
            base_url="https://api.deepseek.com/v1",
        )

    # default: local Ollama
    model = env.get("OLLAMA_MODEL", "qwen2.5:14b")
    return OllamaBackend(model_name=model)


BACKEND = build_backend()
RECOGNIZER = get_recognizer()

print(f"[server] LLM backend: {BACKEND.model_name}")

# чтобы кнопки не дёргали игру одновременно
LOCK = threading.Lock()

# глобальный флаг остановки авто-игры (ставится кнопкой "Stop")
STOP_FLAG = threading.Event()

# Архитектура (C): состояние колоды ведётся В ПАМЯТИ, а не перечитывается
# с экрана каждый ход. Vision нужен только для НАЧАЛЬНОЙ раздачи (в ней
# каждая колонка имеет ровно одну полностью видимую открытую карту) и для
# сверки-якорей. Перекрытые карты веера из пикселей не распознаются вообще —
# их identity уже известен программе из apply_move.
_board = None           # логическое состояние (Board из rule_engine)
_board_initialized = False

# защита от бесконечного цикла: те же N ходов подряд по ЛОГИЧЕСКОМУ состоянию
# (карты не двигаются в памяти) = клики не долетают до игры -> стоп.
_LAST_BOARD_SNAPSHOT = None
_NO_CHANGE_STREAK = 0
STUCK_AFTER = 3  # столько ходов подряд без изменения стола = стоп

# история логических состояний партии (для детекции циклов "туда-сюда").
# Каждый render(board) снапшот запоминается; ход, ведущий в уже виданное
# состояние, отбрасывается — это убирает бессмысленные перекладывания
# свободной карты между двумя колонками туда и обратно.
_SEEN_STATES = set()

# Защита от "шаттл"-цикла: стопка из N карт перекладывается туда-обратно между
# двумя колонками (7↔6, 7↔6...). render-сравнение это НЕ ловит, потому что
# каждый такой ход открывает новую face-down карту и строка состояния меняется.
# Вместо этого запоминаем КЛЮЧ последнего применённого хода tableau→tableau и
# отбрасываем ход, который его точь-в-точь отменяет (те же from/to, тот же N).
_LAST_T2T_KEY = None

# Счётчик подряд идущих ходов tableau→tableau БЕЗ draw/foundation между ними.
# Если LLM закапывается в "шатание" стопок (несколько перекладываний подряд,
# каждое "открывает карту", но реального прогресса нет) — форсируем draw.
_T2T_STREAK = 0
FORCE_DRAW_AFTER_T2T = 2  # после 2 чистых перекладываний подряд вмешиваемся

# Структурный анти-цикл: ловит ЛЮБУЮ длину цикла перекладывания стопок,
# сравнивая сигнатуру расклада колонок с уже виденными за партию. Стейтфул —
# поэтому живёт здесь (в сервере), а не в чистом rule_engine.
_LOOP_GUARD = LoopGuard()

# Живая калибровка шага веера (px) по подтверждённым одиночным ходам. Пока
# нет наблюдений — quantity-check пропускается (не гадаем).
_FAN_STEP = FanStepEstimator()


def _t2t_key(move) -> tuple | None:
    """Ключ хода tableau→tableau для детекции немедленного обратного повтора."""
    if move.kind != "tableau_to_tableau":
        return None
    return (move.from_index, move.to_index, move.card_count)


def _is_reverse(a, b) -> bool:
    """Ход b отменяет ход a (та же стопка, обратные from/to)."""
    return (a is not None and b is not None
            and a[2] == b[2] and a[0] == b[1] and a[1] == b[0])


def foundation_index(suit):
    from orchestrator.rule_engine import REAL_SUITS
    return list(REAL_SUITS).index(suit)


def _fan_step_for_layout(layout):
    """
    Vertical step of the OPEN-card fan (= 5/16 * card_h, ~55px at card_h=176).

    This is the value that governs (a) where to press to lift a stack of N
    open cards and (b) how far a column's free card shifts after a move.
    The closed (blue-back) step (~18px) is a DIFFERENT, unrelated quantity.
    """
    return max(8, int(round(layout.card_h * 0.3125)))


def apply_move_to_screen(layout, move):
    kind = move.kind
    if kind == "draw":
        x, y = layout.stock_xy
        act.click_screen(x, y)
    elif kind == "tableau_to_foundation":
        cx, cy = layout.tableau_card_xy(move.from_index)
        act.click_screen(cx, cy, double=True)
    elif kind == "tableau_to_tableau":
        # Grab the WHOLE run of `card_count` open cards. `open_y` is the top of
        # the BOTTOM (playable) card. To pick up the whole stack we must press
        # on the TOP open card of the run — its top is `open_y` minus
        # (card_count-1) fan steps. Pressing on the bottom card only lifts that
        # single card, which is exactly the "took only the 2♠" bug.
        sx = layout.centers[move.from_index]
        fan_step = _fan_step_for_layout(layout)  # measured (~18px), not ratio
        base_y = layout.open_y[move.from_index] or layout.table_top_y
        sy = base_y - (move.card_count - 1) * fan_step
        # grab slightly BELOW the top card's very top edge to reliably lift it
        sy += fan_step // 2
        tx, ty = layout.tableau_card_xy(move.to_index)
        # Для ПУСТОЙ колонки tableau_card_xy сам отдаёт центр пустого слота:
        # дроп у верхней кромки (table_top_y) игра не принимала.
        act.drag_screen(sx, sy, tx, ty)
    elif kind == "waste_to_foundation":
        x, y = layout.waste_xy
        act.click_screen(x, y, double=True)
    elif kind == "waste_to_tableau":
        sx, sy = layout.waste_xy
        tx, ty = layout.tableau_card_xy(move.to_index)
        act.drag_screen(sx, sy, tx, ty)


def _positions_from_layout(layout):
    """Extract the numeric anchors verify.* needs from a Layout object."""
    return {
        "centers": list(layout.centers),
        "card_w": layout.card_w,
        "card_h": layout.card_h,
        "table_top_y": layout.table_top_y,
        "bottom_y": layout.bottom_y,
    }


def _move_reveals(board, move) -> bool:
    """Does this tableau move uncover a face-down card beneath the moved block?"""
    if move.kind != "tableau_to_tableau":
        return False
    col = board.tableau[move.from_index]
    remaining = col[: len(col) - move.card_count]
    return bool(remaining) and not remaining[-1].face_up


def _shuffle_filter(board, moves):
    """
    Deterministic anti-shuffle: drop tableau→tableau moves that are provably
    "empty" — they reveal no face-down card AND move a card onto an
    EQUIVALENT target (same rank, same color as the current home, just in a
    different column). E.g. Q♠ from one red King onto another red King, or a
    bare King onto an empty column when it frees nothing.

    This is the safety net UNDER the LLM: the model may still misread the
    prompt, but a move that provably gains nothing is removed before the model
    ever sees it. Draw / waste / foundation moves are never filtered.
    """
    kept = []
    for move in moves:
        if move.kind != "tableau_to_tableau":
            kept.append(move)
            continue
        # Reveals are ALWAYS kept — revealing is pure progress.
        if _move_reveals(board, move):
            kept.append(move)
            continue

        src_col = board.tableau[move.from_index]
        moving = src_col[-move.card_count] if src_col else None  # bottom of block
        dst_col = board.tableau[move.to_index]
        target = dst_col[-1] if dst_col else None

        if moving is None or moving.rank == 0:
            kept.append(move)  # unknown card — can't judge, let it through
            continue

        # Target is the SAME rank+color (or empty) as a card the block already
        # sits on → relocation changes nothing except the column number.
        # For a block of N>1, the "anchor" of the block's base is moving (it
        # must be placed on a card one rank higher). Compare by rank+color.
        if target is None:
            # moving onto an empty column: only meaningful for a King, and even
            # then only if it FREES the source (leaves a face-down or empty).
            src_after = src_col[: len(src_col) - move.card_count]
            frees_card = bool(src_after) and not src_after[-1].face_up
            if moving.rank != 13 or frees_card:
                kept.append(move)
            else:
                continue  # bare King parks on empty with no gain — drop
            continue

        # Target card present: relocation is "empty" if target is the same
        # rank AND same color as the card the block's base is moving onto now.
        # Compare base source anchor vs target.
        src_anchor = None
        if len(src_col) > move.card_count:
            src_anchor = src_col[-move.card_count - 1]  # card beneath the block
        if src_anchor is not None and src_anchor.rank == target.rank \
                and src_anchor.suit.is_red == target.suit.is_red:
            continue  # same rank, same color → equivalent spot, drop
        kept.append(move)
    return kept


def _undo():
    """
    Roll the GAME's own state back one move, so a botched drag can be retried.

    Honors UNDO_METHOD from .env:
      - "ctrl_z" (default)   -> send Ctrl+Z via SendInput.
      - "button:x,y"         -> click the game's Undo button at screenshot
                                coords (x, y) — for UWP builds where the hotkey
                                is not wired.

    The game reverts its own board; we do NOT try to reconstruct what the bad
    drag did on screen — the game rolls itself back to the last confirmed state.
    """
    env = read_env()
    method = env.get("UNDO_METHOD", "ctrl_z").strip().lower()
    if method.startswith("button:"):
        try:
            xy = method.split(":", 1)[1]
            x, y = (int(v.strip()) for v in xy.split(","))
            act.click_screen(x, y)
            return
        except Exception:
            pass  # fall through to Ctrl+Z on parse failure
    act.undo_last_move(1)


def _interruptible_sleep(seconds: float, step: float = 0.1) -> bool:
    """
    Sleep in small chunks, checking STOP_FLAG each step. Returns True if the
    user pressed Stop mid-sleep (caller should abort). This replaces plain
    ``time.sleep`` in the move/verify loop so that Stop reacts fast even while
    the code is waiting out an animation — without it, Stop appears to "lag"
    because the handler is blocked inside a single long sleep for up to 1.6s.
    """
    elapsed = 0.0
    while elapsed < seconds:
        if STOP_FLAG.is_set():
            return True
        chunk = min(step, seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk
    return STOP_FLAG.is_set()


def _recognize_one(image, layout, cx, top_y):
    """Recognize a single fully-visible card at (cx, top_y); return (rank, suit)."""
    try:
        res = RECOGNIZER.recognize_at_with_confidence(
            image, cx, max(0, top_y), layout.card_w, layout.card_h
        )
        return res[0], res[1]
    except Exception:
        from orchestrator.rule_engine import UNKNOWN_RANK, Suit
        return UNKNOWN_RANK, Suit.UNKNOWN


def _stock_on_screen(image, layout) -> bool:
    """
    Есть ли на столе колода? Непустой stock рисуется СИНЕЙ рубашкой, пустой —
    зелёным кругом-«recycle» (кольцо на сукне), поэтому признак однозначен:
    пиксель в точке stock_xy не зелёный  <=>  карты в колоде есть.
    """
    x, y = layout.stock_xy
    H, W = image.shape[:2]
    if not (0 <= x < W and 0 <= y < H):
        return False
    r, g, b = (int(v) for v in image[int(y), int(x)][:3])
    return not (g > r + 20 and g > b + 20)


def _sync_stock_presence(image, layout) -> bool:
    """
    Привести СОСТОЯНИЕ колоды в памяти в соответствие с экраном.

    Достоверно с экрана читается только ФАКТ: непустая колода = синяя рубашка,
    пустая = зелёный круг-«recycle». Отсюда две правки памяти:

    1) колода есть на экране  -> в памяти она непуста;
    2) содержимое колоды ВСЕГДА плейсхолдеры. После recycle `apply_move` несёт
       в stock identity из waste — но waste в памяти после рестарта посреди
       партии знает лишь видимый веер (<=3 карт), поэтому эти identity —
       догадка. Если их не обнулить, draw «предсказывает» не ту карту, которая
       реально вскроется (лог: predicted.waste=['*','*','J♣'] при screen=5♦),
       верификация падает, и цикл идёт до Stuck.

    Вызывается и ДО выбора хода (чтобы recycle/draw совпадали с игрой), и после
    подтверждённого хода.
    """
    global _board
    from orchestrator.rule_engine import Card, Suit

    on_screen = _stock_on_screen(image, layout)
    if on_screen:
        if not _board.stock or any(c.rank != 0 for c in _board.stock):
            # реальный размер не важен для выбора хода — важны лишь «непусто»
            # и отсутствие ложных identity
            _board.stock = [Card(0, Suit.UNKNOWN, False) for _ in range(24)]
            return True
        return False
    if _board.stock:
        _board.stock = []
        return True
    return False


def _sync_revealed_card(move, image=None):
    """
    After a move, the ONE card that might have just been REVEALED (drawn from
    the stock, or flipped in a column) is UNKNOWN in the in-memory board. Read
    that single fully-visible card from a fresh screenshot and write it back.

    `image` is the post-move frame already captured by the verification loop;
    when provided, we skip a redundant capture+focus (perf).
    """
    global _board
    from orchestrator.rule_engine import Card, Suit

    if image is None:
        hwnd = act.find_game_window()
        # фокус здесь НЕ нужен: capture_window (mss) читает кадр по координатам
        # окна независимо от фокуса. Лишний SetForegroundWindow на каждом ходе
        # агрессивно перебрасывал фокус и сворачивал другие окна.
        image = act.capture_window(hwnd)
    layout = Layout(image)

    # фактическое наличие/отсутствие колоды — синхронизируем на каждом ходу
    _sync_stock_presence(image, layout)

    # Waste: after a draw (top card just revealed) OR after a waste_to_* move
    # (the card below the played one becomes the new top), the TOP waste card
    # is fully visible and its identity may still be UNKNOWN. Resolve it from
    # the screen's rightmost-waste anchor. Lower fan cards stay UNKNOWN — we
    # only identify a card at the exact moment it becomes the (fully visible)
    # top, never by reading a partially-covered card.
    if move.kind in ("draw", "waste_to_tableau", "waste_to_foundation"):
        if _board.waste:
            # Верхняя waste-карта ВСЕГДА полностью видна -> экран авторитетен.
            # Раньше значение переписывалось ТОЛЬКО когда в памяти стоял
            # UNKNOWN-плейсхолдер. После рестарта посреди партии память знает
            # лишь видимый веер (<= 3 карт), а recycle/draw вскрывают карты,
            # которых она не знает -> «waste: mem=J♣ screen=5♦» и петля до Stuck.
            wx, _wy = layout.waste_xy
            rank, suit = _recognize_one(image, layout, wx, layout.waste_top_y)
            if suit != Suit.UNKNOWN:
                _board.waste[-1] = Card(rank, suit, True)
        return

    if move.kind in ("tableau_to_foundation", "tableau_to_tableau"):
        src = move.from_index
        col = _board.tableau[src]
        # apply_move(_flip_new_top) flipped the top card to face_up=True but
        # left its rank/suit UNKNOWN (it was face-down before). Detect that and
        # read the now-visible card from the screen.
        if col and col[-1].face_up and col[-1].rank == 0:
            cx = layout.centers[src]
            top_y = layout.open_y[src]
            if top_y is not None:
                rank, suit = _recognize_one(image, layout, cx, top_y)
                col[-1] = Card(rank, suit, True)
        return


def _verify_sync():
    """
    Anchor check (architecture C). After a move, re-read the FULLY-VISIBLE
    anchors from a fresh screenshot and compare them against the in-memory
    board: each column's top open card, the waste top, the foundation tops.
    If they diverge, the click did not land and the real game has drifted, so
    the caller rebuilds the board from the screen.

    The comparison is SYMMETRIC on waste: we detect both "a card appeared that
    memory doesn't know" and "the card differs from memory" — a one-sided check
    missed the case where memory thought the waste was empty while a card was
    actually playable, which caused missed waste->tableau moves.
    """
    global _board

    hwnd = act.find_game_window()
    # фокус НЕ нужен для чтения кадра — см. _sync_revealed_card.
    image = act.capture_window(hwnd)
    refr = screen_to_board(image, recognizer=RECOGNIZER)

    # 1) tableau top open cards
    for ci in range(7):
        mem_col = _board.tableau[ci] if ci < len(_board.tableau) else []
        mem_top = mem_col[-1] if mem_col else None
        ref_col = refr.tableau[ci] if ci < len(refr.tableau) else []
        ref_top = ref_col[-1] if ref_col else None
        if mem_top is None and ref_top is None:
            continue
        if mem_top is None or ref_top is None:
            return False
        if mem_top.rank != ref_top.rank or mem_top.suit != ref_top.suit:
            return False

    # 2) waste (symmetric)
    mem_w = _board.waste[-1] if _board.waste else None
    ref_w = refr.waste[-1] if refr.waste else None
    if mem_w is None and ref_w is None:
        pass
    elif mem_w is None or ref_w is None:
        return False
    elif mem_w.rank != ref_w.rank or mem_w.suit != ref_w.suit:
        return False

    return True


def _board_has_duplicates(board) -> bool:
    """
    A board is invalid if the SAME concrete card (rank+suit) appears more than
    once across the whole table (columns + waste + foundations). Face-down
    (UNKNOWN) cards are skipped. After a botched click, apply_move can pile a
    duplicate into a column (e.g. a second 4♠), which then poisons every later
    move — this detects that corruption deterministically, without vision.
    """
    from orchestrator.rule_engine import Suit
    seen = set()
    for col in board.tableau:
        for c in col:
            if c.rank == 0 or c.suit == Suit.UNKNOWN:
                continue
            key = (c.rank, c.suit)
            if key in seen:
                return True
            seen.add(key)
    for c in board.waste:
        if c.rank != 0 and c.suit != Suit.UNKNOWN:
            key = (c.rank, c.suit)
            if key in seen:
                return True
            seen.add(key)
    for suit, pile in board.foundation.items():
        for c in pile:
            if c.rank != 0 and c.suit != Suit.UNKNOWN:
                key = (c.rank, c.suit)
                if key in seen:
                    return True
                seen.add(key)
    return False


def _merge_resync(old_board, fresh_board):
    """
    Resynchronize the in-memory board from a fresh screen read WITHOUT losing
    the BURIED open cards. `fresh_board` (from screen_to_board) only knows each
    column's TOP open card and the count of face-down cards — it cannot read
    open cards overlapped by others (they are partially hidden). Those buried
    open cards are exactly the ones we ALREADY know from `old_board` (apply_move
    tracked them). So we keep the buried open cards from memory and only replace
    the parts the screen can authoritatively report: face-down count, top open
    card, waste, foundations.
    """
    from orchestrator.rule_engine import Card, Suit

    b = old_board.clone()

    # rebuild each tableau column: face-down placeholders + (buried open from
    # memory) + (fresh top open card)
    for ci in range(7):
        old_col = old_board.tableau[ci]
        new_col = fresh_board.tableau[ci]
        # Экран говорит "колонка ПУСТА" — значит веер реально уехал целиком
        # (например, drag сработал, а верификация упала по другой причине).
        # В этом случае НЕ воскрешаем buried-карты из памяти: иначе в пустой
        # колонке появлялись карты-призраки.
        if not new_col:
            b.tableau[ci] = []
            continue
        # face-down count from the fresh read (screen is authoritative here)
        closed = 0
        for c in new_col:
            if not c.face_up:
                closed += 1
            else:
                break
        # top open card from fresh read
        new_top = new_col[-1] if new_col and new_col[-1].face_up else None
        # buried open cards from memory (all open cards except the last one)
        old_open = [c for c in old_col if c.face_up]
        buried = old_open[:-1] if len(old_open) >= 1 else []

        rebuilt = []
        for _ in range(closed):
            rebuilt.append(Card(0, Suit.UNKNOWN, face_up=False))
        rebuilt.extend(buried)
        if new_top is not None:
            # de-dup: drop the fresh top if it already sits in the buried list
            # (this is how a duplicate 4♣ got "remembered" and shuffled around)
            if rebuilt and rebuilt[-1].rank == new_top.rank and rebuilt[-1].suit == new_top.suit:
                pass
            else:
                rebuilt.append(new_top)
        b.tableau[ci] = rebuilt

    # waste: SAME principle as tableau — the screen can only authoritatively
    # report the RIGHTMOST (topmost, playable) waste card; every card to its
    # left in the fan is buried and must be kept from memory, never replaced
    # by `fresh_board.waste` (which screen_to_board always returns as length
    # <= 1). Previously this line was `b.waste = list(fresh_board.waste)`,
    # which silently truncated the whole waste history on every resync —
    # this was the actual cause of "mem=10♥ screen=Q♣": Q♣, and everything
    # drawn after the last resync, was simply erased.
    old_waste = old_board.waste
    new_waste_top = fresh_board.waste[-1] if fresh_board.waste else None
    buried_waste = old_waste[:-1] if old_waste else []
    rebuilt_waste = list(buried_waste)
    if new_waste_top is not None:
        if rebuilt_waste and rebuilt_waste[-1].rank == new_waste_top.rank \
                and rebuilt_waste[-1].suit == new_waste_top.suit:
            pass  # fresh top is already the last buried card — don't duplicate
        else:
            rebuilt_waste.append(new_waste_top)
    b.waste = rebuilt_waste

    b.foundation = {s: list(c) for s, c in fresh_board.foundation.items()}
    b.stock = list(fresh_board.stock)
    return b


def _prevalidate_move_identity(board, move, layout, image):
    """
    Before the physical drag, re-recognize the VISIBLE source/target cards the
    move touches and compare with memory. If memory disagrees with the screen,
    FIX memory right here (write the screen truth) and report a mismatch — the
    caller should NOT perform the drag, and instead re-plan from the corrected
    state.

    This is the expert's key fix: the 16/19 unhealed verification failures are
    identity mismatches (mem≠screen) that poison memory *before* the drag ever
    lands. Checking the 1-2 fully-visible cards just before the move closes
    that class at the root, instead of discovering it after 3 useless retries.

    Returns a dict: {"ok": bool, "reason": str}. On ok=False the board's
    relevant card(s) have already been corrected from the screen.
    """
    from orchestrator.rule_engine import Card, Suit

    def _read(pos, cx, top_y):
        if top_y is None:
            return None
        rank, suit = _recognize_one(image, layout, cx, top_y)
        if suit == Suit.UNKNOWN:
            return None
        return Card(rank, suit, True)

    # 1) waste top (source of any waste_to_* move; also what "draw" reveals)
    if move.kind in ("waste_to_tableau", "waste_to_foundation", "draw"):
        if board.waste:
            wx, wy = layout.waste_xy
            top_y = layout.waste_top_y  # точный верх верхней карты веера
            real = _read("waste", wx, top_y)
            mem = board.waste[-1]
            if real is not None and mem.rank != 0 and (real.rank != mem.rank or real.suit != mem.suit):
                # память расходится с экраном — чиним её точечно
                board.waste[-1] = real
                return {"ok": False, "reason": f"waste mem={mem} screen={real}"}

    # 2) target tableau column top (for waste_to_tableau / tableau_to_tableau)
    if move.kind in ("waste_to_tableau", "tableau_to_tableau"):
        dst = move.to_index
        target_mem = board.tableau[dst][-1] if board.tableau[dst] else None
        # целевая верхняя открытая карта (полностью видима)
        cy = layout.open_y[dst]
        if cy is not None:
            cx = layout.centers[dst]
            real = _read(f"col{dst+1}", cx, cy)
            if real is not None:
                if target_mem is None:
                    # память думала колонка пуста, но на экране есть карта
                    board.tableau[dst].append(real)
                    return {"ok": False, "reason": f"col{dst+1} mem=empty screen={real}"}
                if real.rank != target_mem.rank or real.suit != target_mem.suit:
                    board.tableau[dst][-1] = real
                    return {"ok": False, "reason": f"col{dst+1} mem={target_mem} screen={real}"}

    # 3) source tableau top card being moved (tableau_to_tableau / _foundation)
    if move.kind in ("tableau_to_tableau", "tableau_to_foundation"):
        src = move.from_index
        src_mem = board.tableau[src][-1] if board.tableau[src] else None
        cy = layout.open_y[src]
        if cy is not None:
            cx = layout.centers[src]
            real = _read(f"col{src+1}", cx, cy)
            if real is not None and src_mem is not None and \
                    (real.rank != src_mem.rank or real.suit != src_mem.suit):
                board.tableau[src][-1] = real
                return {"ok": False, "reason": f"col{src+1} mem={src_mem} screen={real}"}

    return {"ok": True, "reason": "ok"}


def _point_fix_identity(board, reason, layout, image):
    """
    After a verification failure whose reason names a column (e.g.
    "col 3 top: mem=10♣ screen=J♦"), re-read THAT column's top card from the
    live screen and write the screen truth directly into memory — instead of
    retrying the same move. Parses the column number from `reason`.

    This is the expert's second fix: the recovery step after a failed verify
    was writing wrong guesses (10♣) instead of the real card (J♦). Point-fix
    writes the correct value to the exact location memory got wrong.
    """
    import re

    from orchestrator.rule_engine import REAL_SUITS, Card, Suit

    # 1) foundation несовпадение: "foundation ♥: mem=A♥ screen=None"
    fm = re.search(r"foundation ([♥♦♣♠])", reason)
    if fm:
        suit_char = fm.group(1)
        suit = next((s for s in REAL_SUITS if s.value == suit_char), None)
        if suit is None:
            return False
        screen_str = re.search(r"screen=(\S+)", reason)
        scr = screen_str.group(1) if screen_str else "None"
        if scr == "None":
            # На экране этот фундамент пуст, а память ждёт карту. Это НЕ
            # точечная починка — обнуление и так пустого слота ничего не
            # даёт (карта-источник осталась "вынутой" в predicted, а в
            # реальности не долетела). Притворяться success нельзя: это
            # загоняет LLM в вечный цикл "туз на фундамент" -> verify fail
            # -> тот же ход. Возвращаем False, чтобы вызывающий код выполнил
            # ПОЛНЫЙ ресинк _board со свежего экрана (foundation и верхние
            # карты колонок здесь полностью видимы -> экран авторитетен).
            return False
        # screen=есть карта: перечитаем её с экрана точечно и запишем
        # (foundation-слоты полностью видны, поэтому просто читаем по xy)
        # координаты foundation-слота: foundation_xy идёт в порядке ♥♦♣♠
        try:
            fi = list(REAL_SUITS).index(suit)
            fx, ftop = layout.foundation_xy[fi]
            rank, s2 = _recognize_one(image, layout, fx, ftop)
            if s2 != Suit.UNKNOWN and s2 == suit:
                board.foundation[suit] = [Card(rank, s2, True)]
                return True
        except Exception:
            pass
        return False

    # 2) waste несовпадение
    if re.search(r"^waste:", reason):
        # Экран авторитетен для ВЕРХНЕЙ карты waste — она всегда полностью
        # видна. Раньше здесь была заглушка: возвращали True, НИЧЕГО не записав
        # в память. Память оставалась неверной, LLM на следующем /move выбирала
        # тот же draw, и цикл крутился до Stuck.
        scr_m = re.search(r"screen=(\S+)", reason)
        scr = scr_m.group(1) if scr_m else "None"
        if scr == "None":
            # на экране waste пуст — значит веер реально пуст
            board.waste = []
            return True
        wx, _wy = layout.waste_xy
        rank, suit = _recognize_one(image, layout, wx, layout.waste_top_y)
        if suit == Suit.UNKNOWN:
            return False
        card = Card(rank, suit, True)
        if board.waste:
            board.waste[-1] = card
        else:
            board.waste.append(card)
        return True

    # 3) tableau column mismatch: "col N top: ..."
    m = re.search(r"col (\d+)", reason)
    if not m:
        return False
    col_idx = int(m.group(1)) - 1  # reason uses 1-based column numbers
    if not (0 <= col_idx < 7):
        return False

    cy = layout.open_y[col_idx]
    if cy is None:
        return False
    cx = layout.centers[col_idx]
    rank, suit = _recognize_one(image, layout, cx, cy)
    if suit == Suit.UNKNOWN:
        return False

    col = board.tableau[col_idx]
    if col and col[-1].face_up:
        col[-1] = Card(rank, suit, True)
    elif col:
        col.append(Card(rank, suit, True))
    else:
        col.append(Card(rank, suit, True))
    return True


def _looks_like_fresh_deal(fresh_board) -> bool:
    """Heuristic: does `fresh_board` look like a brand-new Klondike deal?

    A fresh deal has: empty waste, no foundation cards, and each tableau column
    a fan of exactly (index+1) cards — i.e. closed counts 1,2,3,4,5,6 with a
    single open card on top. This is used to DETECT that the user started a new
    game in the UI while the server still holds memory of the previous game.
    """
    if fresh_board.waste or any(fresh_board.foundation.values()):
        return False
    closed_by_col = []
    for col in fresh_board.tableau:
        closed = sum(1 for c in col if not c.face_up)
        closed_by_col.append(closed)
    return closed_by_col == [0, 1, 2, 3, 4, 5, 6]


def do_one_move():
    """One move: (memory Board) -> LLM -> Act -> apply_move (memory)."""

    global _board, _board_initialized, _LAST_BOARD_SNAPSHOT, _NO_CHANGE_STREAK, _LAST_T2T_KEY, _T2T_STREAK

    with LOCK:
        # РАННЯЯ проверка Stop: если авто-цикл уже успел поставить несколько
        # /move запросов в очередь (они ждут LOCK), то после нажатия Stop они
        # не должны выполняться. Иначе «игра прокручивается ещё 1-2 хода»
        # после остановки.
        if STOP_FLAG.is_set():
            return {
                "reasoning": "Stopped.",
                "move": None,
                "model": BACKEND.model_name,
                "state": render(_board) if _board else "",
                "won": False,
            }

        hwnd = act.find_game_window()
        # фокус убран отсюда: чтение доски не требует фокуса, а ранний
        # SetForegroundWindow + долгий LLM-запрос означали, что к моменту
        # клика фокус снова терялся (мышь/user переключались) -> "Stuck".
        # Фокус теперь берём ОДИН раз, прямо перед физическим кликом.
        try:
            image = act.capture_window(hwnd)
        except RuntimeError as e:
            # окно свёрнуто/скрыто (capture_window теперь явно сигналит об
            # этом вместо мусорного mss.ScreenShotError -32000). Честно
            # сообщаем, а НЕ крутим ход по устаревшей памяти.
            return {
                "reasoning": f"Cannot read the game window: {e}",
                "move": None,
                "model": BACKEND.model_name,
                "state": render(_board) if _board else "",
                "won": False,
                "stuck": False,
            }

        # --- build the initial board ONCE (fresh deal) ---------------------
        # In a fresh deal every column has exactly ONE fully-visible open card,
        # so screen_to_board reads them reliably (conf ~1.0). After that we
        # carry the Board forward in memory via apply_move and never re-read
        # partially-overlapped cards from pixels.
        try:
            if not _board_initialized:
                _board = screen_to_board(image, recognizer=RECOGNIZER)
                _board_initialized = True
                _LAST_BOARD_SNAPSHOT = None
                _NO_CHANGE_STREAK = 0
            else:
                # Detect a NEW GAME started in the UI while we still carry the
                # previous game's memory. If the screen now shows a fresh deal
                # (and the memory does not), drop the memory and re-read.
                refr = screen_to_board(image, recognizer=RECOGNIZER)
                if _looks_like_fresh_deal(refr) and not _looks_like_fresh_deal(_board):
                    _board = refr
                    _SEEN_STATES.clear()
                    _LAST_T2T_KEY = None
                    _T2T_STREAK = 0
                    _LOOP_GUARD.reset_progress()
                    _log_move("detected NEW GAME (fresh deal) — memory reset from screen")
        except (ValueError, IndexError) as e:
            # стол не виден (окно свёрнуто / белый фон / вне кадра) — не роняем
            # /move, а честно сообщаем.
            return {
                "reasoning": f"Board not readable (table not visible?): {e}",
                "move": None,
                "model": BACKEND.model_name,
                "state": render(_board) if _board else "",
                "won": False,
            }

        # ── ЖИВОЙ кадр авторитетен: если стол сейчас НЕ читается (меню/пауза/
        # реклама/свёрнутое окно), НЕЛЬЗЯ генерировать ход по устаревшей памяти —
        # "железное правило туза" иначе вечно выбирает Move ... to foundation из
        # застывшего _board, клики уходят в неигровой экран, верификация даёт
        # screen=None, и цикл крутится до Stuck. Проверяем ЧИТАЕМОСТЬ доски прямо
        # здесь, до любого выбора хода, и честно выходим, если стол не виден.
        # Исключение ValueError от find_tableau_columns ("table not visible") и
        # есть детектор: нет 7 колонок — значит на экране не стол.
        try:
            _fresh_probe = screen_to_board(image, recognizer=RECOGNIZER)
        except (ValueError, IndexError) as e:
            return {
                "reasoning": (f"Table not visible on screen (menu/pause/minimized?): {e}. "
                              "Start or focus a real deal, then retry."),
                "move": None,
                "model": BACKEND.model_name,
                "state": render(_board) if _board else "",
                "won": False,
                "stuck": False,
            }

        board = _board
        layout = Layout(image)

        # Колода: сверяем память с экраном ДО выбора хода. Иначе память может
        # выбрать recycle (stock пуст) там, где игра физически ждёт обычный
        # draw, и наоборот — ход уходит «не в ту» механику колоды.
        if _sync_stock_presence(image, layout):
            _log_move("stock presence synced from screen before planning")

        # Corruption guard: if the in-memory board holds a duplicate card, the
        # memory is poisoned. Re-sync it from the screen via _merge_resync,
        # which PRESERVES buried open cards (a plain screen_to_board would
        # drop them, "forgetting" the stack and causing the back-and-forth
        # shuffle of a whole pile).
        if _board_has_duplicates(board):
            try:
                refr = screen_to_board(image, recognizer=RECOGNIZER)
                _board = _merge_resync(_board, refr)
                board = _board
            except Exception:
                pass

        if board.is_won():
            return {"reasoning": "Game won! 🏆", "move": None,
                    "state": render(board), "won": True, "model": BACKEND.model_name}

        moves = get_valid_moves(board)
        # Отфильтровать ходы tableau→tableau, ведущие в уже посещённый расклад
        # колонок (структурный анти-цикл — ловит цикл любой длины/формы).
        # draw и ходы к foundation никогда не фильтруются.
        moves = _LOOP_GUARD.filter_progressing_moves(board, moves)
        # Детерминированный анти-шаффл: убрать tableau→tableau ходы, которые
        # ничего не вскрывают и перекладывают карту на эквивалентную цель
        # (тот же ранг+цвет) или паркуют голого короля на пустую колонку без
        # выгоды. Защита ПОД слабым LLM — модель просто не увидит эти ходы.
        moves = _shuffle_filter(board, moves)
        if _LOOP_GUARD.is_stuck():
            return {
                "reasoning": "Stuck: the deck has been recycled several times "
                             "with no progress — this deal appears unsolvable. "
                             "Start a new game.",
                "move": None,
                "model": BACKEND.model_name,
                "state": render(board),
                "won": False,
                "stuck": True,
            }
        if not moves:
            # End-of-stock / recycle edge case: the in-memory board may think
            # the stock AND waste are both empty (green circle visible) while
            # the screen still has a playable waste card or a recyclable stock.
            # Re-read the screen once before declaring "stuck".
            fresh = act.capture_window(hwnd)
            refr = screen_to_board(fresh, recognizer=RECOGNIZER)
            refr_moves = get_valid_moves(refr)
            if refr_moves:
                _board = _merge_resync(_board, refr)
                _board_initialized = True
                board = _board
                moves = refr_moves
            else:
                return {"reasoning": "Stuck — no moves.", "move": None,
                        "state": render(board), "won": False, "model": BACKEND.model_name}

        # Stop-флаг останавливает активный авто-цикл, но НЕ должен блокировать
        # свежий ручной ход. Флаг взводится только кнопкой "Stop" (не Stuck —
        # Stuck возвращается раньше, до этой проверки). Поэтому здесь он уже
        # легитимно взведён только если пользователь САМ нажал Stop — и тогда
        # честно останавливаем. Никакого авто-сброса: залипание флага лечится
        # на клиенте (makeMove теперь шлёт /start перед первым ходом).
        if STOP_FLAG.is_set():
            return {"reasoning": "Stopped.", "move": None,
                    "state": render(board), "won": False, "model": BACKEND.model_name}

        foundation_moves = [m for m in moves
                            if m.kind in ("waste_to_foundation",
                                          "tableau_to_foundation")]

        # ЖЕЛЕЗНОЕ правило: туз ВСЕГДА идёт в фундамент, не спрашивая LLM.
        # Туз на пустую ячейку фундамента — единственный легальный и всегда
        # правильный ход (никакой стратегии: туз нельзя удержать в tableau с
        # выгодой). Раньше мы отдавали это LLM, и она систематически клала туз
        # на двойку в tableau или вообще пропускала второй туз. Теперь: любой
        # ход *_to_foundation, где двигаемая карта — туз (rank 1), берём сразу.
        def _moving_card(m):
            if m.kind == "waste_to_foundation":
                return board.waste[-1] if board.waste else None
            if m.kind == "tableau_to_foundation":
                col = board.tableau[m.from_index]
                return col[-1] if col else None
            return None

        ace_to_foundation = next(
            (m for m in foundation_moves if _moving_card(m) is not None
             and _moving_card(m).rank == 1),
            None,
        )

        # Далее — прочие foundation-ходы выше по приоритету у LLM, но не туз.
        if foundation_moves and all(m.kind in ("waste_to_foundation", "tableau_to_foundation") or m.kind == "draw" for m in moves):
            chosen = foundation_moves[0]
            reasoning = (
                "Card goes to the foundation (the only legal / always-correct move here)."
            )
        else:
            # ВАЖНО: туз в foundation — безусловный приоритет, выше любого
            # другого выбора (включая LLM). Ставим его ПЕРВЫМ.
            if ace_to_foundation is not None:
                chosen = ace_to_foundation
                reasoning = ("Ace goes straight to the foundation (iron rule).")
            else:
                result = choose_move(BACKEND, board, moves)
                chosen = moves[result.move_index]
                reasoning = result.reasoning
                _log_move(f"moves={[repr(m) for m in moves]}")
                _log_move(f"llm_pick={result.move_index} -> {repr(chosen)}")

        if STOP_FLAG.is_set():
            return {
                "reasoning": "Stopped before applying the move.",
                "move": None,
                "model": BACKEND.model_name,
                "state": render(board),
                "won": False,
            }

        # cycle detection: reject a move that leads back to a state we already
        # saw this game (the "shuttle a free card back and forth" bug). Try the
        # other valid moves first; only if ALL lead to seen states, fall back.
        chosen_repr = repr(chosen)
        future = apply_move(board, chosen)
        future_key = render(future)
        if future_key in _SEEN_STATES and moves:
            alt = None
            for cand in moves:
                if cand is chosen:
                    continue
                ckey = render(apply_move(board, cand))
                if ckey not in _SEEN_STATES:
                    alt = cand
                    break
            if alt is not None:
                chosen = alt
                chosen_repr = repr(chosen)
                reasoning = ("(cycle-guard) " + reasoning +
                             f" avoided repeating a seen state; chose {chosen_repr}.")

        # NOTE: cycle/shuffle loops are now handled by the STRUCTURAL LoopGuard
        # (filter_progressing_moves above), which bans any tableau→tableau move
        # that returns the column layout to an already-visited state. The old
        # shuttle-guard / shuffle-streak-guard (heuristics keyed on move SHAPE)
        # were removed — they fired false positives and forced an unnecessary
        # `draw` right after legitimate progress moves (e.g. blocking 4♦→5♠).

        # loop guard on the LOGICAL board: if the in-memory board does not
        # change across N moves, the clicks are not landing and we must stop.
        board_snapshot = render(board)
        if board_snapshot == _LAST_BOARD_SNAPSHOT:
            _NO_CHANGE_STREAK += 1
        else:
            _NO_CHANGE_STREAK = 0
        _LAST_BOARD_SNAPSHOT = board_snapshot

        if _NO_CHANGE_STREAK >= STUCK_AFTER:
            _NO_CHANGE_STREAK = 0
            _LAST_BOARD_SNAPSHOT = None
            return {
                "reasoning": (
                    f"Stopped: the in-memory board did not change for "
                    f"{STUCK_AFTER} moves in a row (last: '{chosen_repr}'). "
                    "Clicks may not be reaching the game (check window focus)."
                ),
                "move": None,
                "model": BACKEND.model_name,
                "state": render(board),
                "won": board.is_won(),
                "stuck": True,
            }

        # ── perform + VERIFY (architecture B: memory must match the screen) ─
        # The physical drag is NOT guaranteed to land. apply_move (memory) is
        # applied ONLY after we confirm the SCREEN now matches the predicted
        # board. We compare the FULLY-VISIBLE anchors only (each column's free
        # card + waste + foundations, all conf ~1.0) — this is deterministic and
        # free of the fan_step ambiguity that broke the earlier shift-check.
        predicted = apply_move(board, chosen)
        _log_move(f"chosen={repr(chosen)}")
        _log_move(f"before.waste={[repr(c) for c in board.waste]} stock={len(board.stock)}")
        _log_move(f"predicted.waste={[repr(c) for c in predicted.waste]}")

        ok = False
        reason = "not verified"
        verify_failures = 0
        point_fixed = False

        # ПРЕД-ПРОВЕРКА identity: перед ПЕРВЫМ drag
        # точечно перечитываем видимые карты-источник и цель, сравниваем с
        # памятью. Если память разошлась с экраном — чиним её и НЕ тащим ход,
        # а возвращаем управление (планировщик перевыберет из исправленного
        # состояния). Это закрывает класс 16/19 identity-провалов ДО хода.
        # Берём СВЕЖИЙ кадр: между входом в do_one_move и этим моментом прошёл
        # LLM-запрос, и `image`/`layout` могли устареть.
        image = act.capture_window(hwnd)
        layout = Layout(image)
        pre = _prevalidate_move_identity(board, chosen, layout, image)
        if not pre["ok"]:
            _log_move(f"PRE-CHECK mismatch: {pre['reason']} — skipping drag, re-plan")
            return {
                "reasoning": (f"(pre-check) memory corrected from screen: "
                              f"{pre['reason']}. Will re-plan."),
                "move": None,
                "model": BACKEND.model_name,
                "state": render(board),
                "won": board.is_won(),
            }

        for attempt in range(3):
            # Stop-проверка прямо перед физическим ходом: если пользователь
            # нажал Stop пока LLM считала — не дёргаем игру зря.
            if STOP_FLAG.is_set():
                return {
                    "reasoning": "Stopped before applying the move.",
                    "move": None,
                    "model": BACKEND.model_name,
                    "state": render(board),
                    "won": False,
                }

            # (re)perform the drag, then pause for the animation to settle.
            # ОДИН раз перед первым drag берём фокус игры: UWP принимает клики
            # только в фокусе. Фокус берём ЗДЕСЬ (прямо перед кликом, после
            # LLM), а не в начале — иначе за долгий LLM-запрос фокус теряется,
            # мышь остаётся "в другом месте" и клик промахивается (Stuck).
            if attempt == 0:
                act.focus_window(hwnd)
                time.sleep(0.4)  # дать игре принять фокус и убрать мышь
            # ПРЕРЫВАЕМЫЙ sleep после хода: Stop срабатывает мгновенно.
            apply_move_to_screen(layout, chosen)
            if _interruptible_sleep(1.6):
                return {
                    "reasoning": "Stopped mid-move.",
                    "move": None,
                    "model": BACKEND.model_name,
                    "state": render(board),
                    "won": False,
                }

            image_after = act.capture_window(hwnd)

            ok, reason = vfy.verify_board_tops(image_after, predicted, recognize=RECOGNIZER)

            # QUANTITY check (destination column) — v2. Проверяем только
            # КОЛОНКУ-ЦЕЛЬ хода (см. quantity_guard docstring: destination
            # даёт чистый N*fan_step, вскрытие рубашки там невозможно).
            new_layout = None
            if ok and chosen.kind in ("tableau_to_tableau", "waste_to_tableau"):
                new_layout = Layout(image_after)
                open_y_before = layout.open_y[chosen.to_index]
                open_y_after = new_layout.open_y[chosen.to_index]
                ok, reason = verify_quantity(
                    open_y_before, open_y_after, chosen.card_count,
                    _FAN_STEP.value, layout.table_top_y,
                )

            _log_move(f"  attempt {attempt}: ok={ok} reason={reason}")
            if ok:
                # калибровка на ЛЮБОМ подтверждённом ходе (переиспользуем
                # уже построенный new_layout, не пересчитываем раскладку).
                if (chosen.kind in ("tableau_to_tableau", "waste_to_tableau")
                        and new_layout is not None):
                    ob = layout.open_y[chosen.to_index]
                    oa = new_layout.open_y[chosen.to_index]
                    if ob is not None and oa is not None:
                        _FAN_STEP.observe(abs(oa - ob), chosen.card_count)
                break

            verify_failures += 1

            # При identity-несовпадении (mem≠screen)
            # НЕ гнать тот же ход ещё 2 раза (79% таких ретраев бесполезны),
            # а сразу точечно перечитать имя-позицию из reason и записать
            # правильное значение в память, затем прервать retry и дать
            # планировщику выбрать заново из исправленного состояния.
            if attempt == 0 and ("mem=" in reason and "screen=" in reason):
                fixed = _point_fix_identity(board, reason, layout, image_after)
                if fixed:
                    _log_move("  identity mismatch — point-fixed from screen, re-plan")
                    point_fixed = True
                    break
                else:
                    # Фикс НЕ удался (foundation-провал вернул False, либо карта
                    # не читается / open_y=None). Просто "give up this move" и
                    # вернуть управление НЕДОСТАТОЧНО: память остаётся кривой
                    # (туз "вынут" в predicted, а в _board не долетел физически),
                    # и LLM на следующем /move снова выберет тот же ход — цикл
                    # не разрывается, а лишь становится молчаливым.
                    #
                    # Правильно: ПОЛНЫЙ ресинк _board со свежего экрана ДО
                    # возврата управления. В этом сценарии foundation и верхние
                    # открытые карты колонок полностью видимы (conf ~1.0), поэтому
                    # экран авторитетен, а _merge_resync сохранит buried open
                    # карты, которые экран не может прочитать (перекрытые).
                    _log_move(f"  identity fix FAILED ({reason}) — full board resync")
                    try:
                        rb = act.capture_window(hwnd)
                        refr = screen_to_board(rb, recognizer=RECOGNIZER)
                        _board = _merge_resync(_board, refr)
                        _board_initialized = True
                        _SEEN_STATES.add(render(_board))
                    except Exception as e:
                        _log_move(f"  resync failed: {e}")
                    return {
                        "reasoning": (f"Move unconfirmed ({reason}); board "
                                      f"re-read from screen. Re-planning."),
                        "move": None,
                        "model": BACKEND.model_name,
                        "state": render(_board),
                        "won": _board.is_won(),
                        "stuck": False,
                    }

            # roll the screen back via Undo, then re-measure before retrying.
            # ПРЕРЫВАЕМЫЙ sleep — иначе после Stop ждём лишнюю секунду и
            # продолжаем дёргать игру.
            _undo()
            if _interruptible_sleep(1.0):
                return {
                    "reasoning": "Stopped mid-move.",
                    "move": None,
                    "model": BACKEND.model_name,
                    "state": render(board),
                    "won": False,
                }
            rb = act.capture_window(hwnd)
            layout = Layout(rb)

        if ok:
            _board = predicted
            _SEEN_STATES.add(render(_board))
            _sync_revealed_card(chosen, image_after)
            _LAST_T2T_KEY = _t2t_key(chosen)  # remember for the shuttle-guard
            # Ход на foundation — необратимый прогресс: разрешаем заново
            # проходить прежние расклады колонок.
            if chosen.kind in ("tableau_to_foundation", "waste_to_foundation"):
                _LOOP_GUARD.reset_progress()
            # Реальный прогресс без foundation тоже сбрасывает счётчик "тупика":
            # карта ушла с waste на стол (waste_to_tableau) или tableau-ход
            # вскрыл face-down карту (раскрытие = информация и новые опции).
            # Иначе recycle счётчик копит "stalled" даже при живых ходах и
            # ложно объявляет нерешаемость (ложный Stuck на проходе 3).
            if chosen.kind == "waste_to_tableau":
                _LOOP_GUARD.reset_progress()
            elif chosen.kind == "tableau_to_tableau" and _move_reveals(board, chosen):
                _LOOP_GUARD.reset_progress()
            # Recycle (draw при пустом stock) без прогресса — счётчик "тупика".
            if chosen.kind == "draw" and len(board.stock) == 0:
                _LOOP_GUARD.note_recycle_without_progress()
        else:
            # ТОЧЕЧНЫЙ фикс identity уже записал правильную карту в `board`
            # (см. _point_fix_identity выше). Синхронизируем глобальную память
            # с исправленным board и возвращаемся — НЕ делаем _merge_resync,
            # который мог бы перезаписать точечную правку (и сам был причиной
            # «10♣ вместо J♦»).
            if point_fixed:
                _board = board
                _SEEN_STATES.add(render(_board))
                return {
                    "reasoning": (reasoning + f" (memory point-fixed from screen: "
                                  f"{reason}; will re-plan)"),
                    "move": None,
                    "model": BACKEND.model_name,
                    "state": render(_board),
                    "won": _board.is_won(),
                    "stuck": False,
                }

            # The drag could not be confirmed and Undo may itself be unreliable
            # in this UWP build. Rather than get stuck with a drifted screen, we
            # re-read the AUTHORITATIVE fully-visible state from the screen and
            # accept it (this loses buried cards only if the drag truly botched,
            # but keeps the game moving instead of shuttling forever).
            try:
                rb = act.capture_window(hwnd)
                refr = screen_to_board(rb, recognizer=RECOGNIZER)
                # _merge_resync, NOT a raw screen_to_board assignment — the raw
                # read cannot represent buried tableau cards or waste history
                # (see its own docstring), so using it directly here was the
                # actual cause of lost buried cards and waste desync.
                _board = _merge_resync(_board, refr)
                _SEEN_STATES.add(render(_board))
                reasoning += (f" (move unconfirmed after {verify_failures} tries — "
                              "board re-read from screen)")
            except Exception:
                reasoning += f" (verification failed: {reason})"
            return {
                "reasoning": reasoning,
                "move": chosen_repr,
                "model": BACKEND.model_name,
                "state": render(_board),
                "won": _board.is_won(),
                # ВАЖНО: провал верификации ≠ партия нерешаема. Это значит
                # "этот ход не подтвердился, доска перечитана с экрана,
                # продолжаем". Если поставить stuck=True, клиент навсегда
                # остановит автоигру — из-за одного сорванного drag'а.
                # stuck резервируем ТОЛЬКО под честный тупик (нет ходов /
                # recycle-цикл без прогресса).
                "stuck": False,
            }

        return {
            "reasoning": reasoning,
            "move": chosen_repr,
            "model": BACKEND.model_name,
            "state": render(_board),
            "won": _board.is_won(),
        }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (_ROOT / "web" / "index.html").read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/dump":
            self._send_json({"board": render(_board) if _board else "(no board)",
                             "init": _board_initialized,
                             "seen_count": len(_SEEN_STATES)})
        else:
            self._send_json({"error": "not found"})

    def do_POST(self):
        if self.path == "/move":
            try:
                result = do_one_move()
                self._send_json(result)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                try:
                    (_ROOT / "server_error.log").write_text(tb, encoding="utf-8")
                except Exception:
                    pass
                print("[server] /move error:\n" + tb, flush=True)
                self._send_json({"reasoning": f"Error: {e}", "move": None,
                                 "state": "", "won": False, "traceback": tb})
        elif self.path == "/stop":
            STOP_FLAG.set()
            self._send_json({"reasoning": "Stopped.", "move": None,
                             "state": "", "won": False})
        elif self.path == "/shutdown":
            # hard-kill the server process from inside, freeing port 8000 so a
            # fresh start.bat never collides with a zombie server.
            self._send_json({"reasoning": "Shutting down server...", "move": None,
                             "state": "", "won": False})
            self.wfile.flush()
            import os
            os._exit(0)
        elif self.path == "/start":
            STOP_FLAG.clear()
            # start a NEW game: drop the in-memory board so the next move
            # re-reads the fresh deal from the screen, and reset the guard
            # and the cycle-history.
            global _board, _board_initialized, _LAST_BOARD_SNAPSHOT, _NO_CHANGE_STREAK, _SEEN_STATES, _LAST_T2T_KEY, _T2T_STREAK
            _board = None
            _board_initialized = False
            _LAST_BOARD_SNAPSHOT = None
            _NO_CHANGE_STREAK = 0
            _SEEN_STATES = set()
            _LAST_T2T_KEY = None
            _T2T_STREAK = 0
            _LOOP_GUARD.reset_progress()
            self._send_json({"reasoning": "Started.", "move": None,
                             "state": "", "won": False})
        else:
            self._send_json({"error": "not found"})

    def _send_json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # the browser aborted the request (Stop button / tab closed / long
            # auto-play loop). The move already happened server-side — just
            # drop the socket, don't crash the handler thread.
            pass

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = 8000
    # Bind on 127.0.0.1 (IPv4). Note: this does NOT listen on ::1 (IPv6), so
    # the browser URL must use 127.0.0.1 (not "localhost", which can resolve to
    # ::1 on some machines and then fail to connect).
    print(f"Server running: http://127.0.0.1:{port}")
    print("Open it in your browser. Stop: Ctrl+C.")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
