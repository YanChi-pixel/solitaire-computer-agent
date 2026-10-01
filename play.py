"""
Orchestrator — glues all layers into a "see → think → move" loop.

One call = one move:
1. screenshot the game window (Act);
2. build Board from the screenshot (bridge);
3. valid moves (rule_engine);
4. LLM picks a move + reasoning (planner);
5. translate the move into pixels and perform it with the mouse (Act).

Usage:  python play.py
"""

import sys
import time
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
from orchestrator.rule_engine import (
    REAL_SUITS,
    Suit,
    get_valid_moves,
    render,
)
from planner.llm_planner import OpenAICompatibleBackend, choose_move
from vision.recognizer import get_recognizer
from vision.screen_to_board import screen_to_board


def read_api_key() -> str:
    for line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("DEEPSEEK_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("DEEPSEEK_API_KEY not found in .env")


def build_backend():
    return OpenAICompatibleBackend(
        model_name="deepseek-chat",
        api_key=read_api_key(),
        base_url="https://api.deepseek.com/v1",
    )


def foundation_index(suit: Suit) -> int:
    """Order index of a foundation pile for a suit (0..3)."""
    return list(REAL_SUITS).index(suit)


def apply_move_to_screen(layout: Layout, move) -> None:
    """Translate a logical Move into real mouse clicks."""
    kind = move.kind

    if kind == "draw":
        # click the stock pile
        x, y = layout.stock_xy
        act.click_screen(x, y)
        return

    if kind == "tableau_to_foundation":
        # double-click the top face-up card (auto-moves to foundation)
        cx, cy = layout.tableau_card_xy(move.from_index)
        act.click_screen(cx, cy, double=True)
        return

    if kind == "tableau_to_tableau":
        sx, sy = layout.tableau_card_xy(move.from_index)
        tx, ty = layout.tableau_card_xy(move.to_index)
        # if the target column is empty, drag to the empty spot
        if layout.open_y[move.to_index] is None:
            ty = layout.table_top_y
        act.drag_screen(sx, sy, tx, ty)
        return

    if kind == "waste_to_foundation":
        x, y = layout.waste_xy
        act.click_screen(x, y, double=True)
        return

    if kind == "waste_to_tableau":
        sx, sy = layout.waste_xy
        tx, ty = layout.tableau_card_xy(move.to_index)
        if layout.open_y[move.to_index] is None:
            ty = layout.table_top_y
        act.drag_screen(sx, sy, tx, ty)
        return

    raise ValueError(f"Unknown move kind: {move.kind}")


def one_turn(backend, recognizer) -> None:
    """One full turn: see -> think -> move."""
    hwnd = act.find_game_window()
    act.focus_window(hwnd)
    image = act.capture_window(hwnd)

    board = screen_to_board(image, recognizer=recognizer)
    layout = Layout(image)

    print(render(board))
    moves = get_valid_moves(board)
    if board.is_won():
        print("WIN!")
        return
    if not moves:
        print("Stuck — no moves.")
        return

    result = choose_move(backend, board, moves)
    chosen = moves[result.move_index]

    print(f"\nReasoning: {result.reasoning}")
    print(f"Move: {chosen}")

    apply_move_to_screen(layout, chosen)
    print("Move performed on screen")
    time.sleep(1.5)  # let the game animate the move


def main():
    backend = build_backend()
    recognizer = get_recognizer()
    one_turn(backend, recognizer)


if __name__ == "__main__":
    main()
