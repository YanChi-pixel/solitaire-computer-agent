"""
Prompt contract for the LLM strategist.

Principle: the model NEVER invents a move from scratch. It receives a numbered
list of valid moves already computed by the rule engine, and its only job is
to pick one index and explain why. This way the model physically cannot make
a nonexistent move.
"""

from __future__ import annotations

import json
import re

from orchestrator.rule_engine import Board, Move, render

SYSTEM_PROMPT = """You are a strategist in Klondike solitaire.
You are given the current board state and a numbered list of VALID moves.
Your job is to pick ONE move by its number and briefly explain why.

HARD RULES (follow them strictly, in this order):
1. If any move sends a card to a foundation, pick it first — never consider
   other moves while a foundation move is available.
2. Revealing a FACE-DOWN card in the tableau is your TOP priority after
   foundation. A move that uncovers a hidden card (e.g. moving the last card
   off a pile that has a face-down card beneath it, or moving a multi-card
   stack off to expose the card under it) is worth more than almost anything
   else. Prefer the tableau move that reveals a face-down card.
3. Only AFTER checking for reveals: if the top waste card can be placed onto
   the tableau (onto a King, or onto a matching lower card), consider it — but
   a waste move that reveals nothing is WEAKER than a tableau move that reveals
   a face-down card.
4. NEVER make a move you could immediately undo for no gain. A shuffle move
   (moving a stack one step and being able to move it right back) is worse
   than drawing.
4b. NEVER shift a stack between two columns that end in the SAME card (two
   identical target cards, e.g. two black 4s, two red 7s). Moving a stack from
   one black 4 onto another black 4 is not progress — it is a pointless shuffle
   that just trades one equivalent spot for another. Only move a stack if its
   NEW home is genuinely different (different rank/color, or reveals a truly
   useful card).
5. Do NOT move a King to an empty column UNLESS that King is followed by a
   run of stacked cards you want to move OFF it, or unless the empty column is
   the ONLY way to free a useful card. Moving a bare King (or a King with a
   useless single card) to an empty column just to "park" it is a waste — it
   reveals nothing and frees nothing. A King already on a good column should
   stay put.
6. Do NOT keep shuffling the SAME cards back and forth between columns (e.g.
   Q♠ from one red King to another, then back). If a move does not reveal a
   face-down card and does not place waste and does not build the foundation,
   it is a shuffle — prefer drawing the stock over shuffling.
7. Draw from the stock as soon as none of your moves reveals a face-down card,
   builds the foundation, or moves waste to the tableau. Do NOT get stuck
   loop-shuffling tableau cards while the stock still holds cards; drawing is
   how you find new options.

Key idea: uncovering face-down cards is how you win Klondike. Every face-down
card you reveal is pure progress. A waste card you can always draw later in a
cycle; a face-down card revealed now is information and options you did not
have before. So: reveal > build-foundation > place-waste > draw > shuffle.

Keep the reasoning SHORT: 1-3 sentences. Do not write long essays.

Response rules:
1. 1-3 sentences of reasoning.
2. At the very end, a SEPARATE line with JSON exactly like:
   {"move_index": <move number from the list>}
3. Do not pick a number not in the list. Do not invent moves.
"""


def _move_reveals(board: Board, move: Move) -> bool:
    """Whether a tableau move uncovers a face-down card beneath the moved stack."""
    if move.kind != "tableau_to_tableau":
        return False
    col = board.tableau[move.from_index]
    # after removing the top `card_count` face-up cards, is the new top face-down?
    remaining = col[: len(col) - move.card_count]
    return bool(remaining) and not remaining[-1].face_up


def _card_label(card) -> str:
    """Short human label for a card; '?' for face-down, '?'+suit-empty for unknown."""
    if card is None:
        return "(empty)"
    r = {1: "A", 11: "J", 12: "Q", 13: "K"}.get(card.rank, str(card.rank))
    s = card.suit.value
    if card.suit.name == "UNKNOWN":
        return "?"
    return f"{r}{s}"


def _moving_top(board: Board, move: Move):
    """Return the bottom card of the block being moved (the card that must land)."""
    if move.kind == "tableau_to_tableau":
        col = board.tableau[move.from_index]
        return col[-move.card_count] if col else None
    if move.kind == "waste_to_tableau" or move.kind == "waste_to_foundation":
        return board.waste[-1] if board.waste else None
    if move.kind == "tableau_to_foundation":
        col = board.tableau[move.from_index]
        return col[-1] if col else None
    return None


def _target_top(board: Board, move: Move):
    """Return the top card of the destination column (None = empty column)."""
    if move.kind in ("tableau_to_tableau", "waste_to_tableau"):
        col = board.tableau[move.to_index]
        return col[-1] if col else None
    return None


def _describe_move(board: Board, move: Move, index: int) -> str:
    # column numbers are 1-based for both the model and the human reader
    if move.kind == "draw":
        top = board.waste[-1] if board.waste else None
        top_s = f" (top of waste: {_card_label(top)})" if top is not None else ""
        return f"{index}. Draw a card from the stock (stock -> waste){top_s}"
    if move.kind == "waste_to_foundation":
        return f"{index}. Move top waste card {_card_label(_moving_top(board, move))} to the foundation"
    if move.kind == "waste_to_tableau":
        empty = " (EMPTY column)" if not board.tableau[move.to_index] else ""
        return f"{index}. Move top waste card {_card_label(_moving_top(board, move))} to column {move.to_index + 1} [onto {_card_label(_target_top(board, move))}]{empty}"
    if move.kind == "tableau_to_foundation":
        return f"{index}. Column {move.from_index + 1}: move {_card_label(_moving_top(board, move))} to the foundation"
    if move.kind == "tableau_to_tableau":
        reveal = "  [REVEALS a hidden card]" if _move_reveals(board, move) else ""
        empty = "  (EMPTY column)" if not board.tableau[move.to_index] else ""
        return (
            f"{index}. Column {move.from_index + 1}: move {_card_label(_moving_top(board, move))} "
            f"({move.card_count} card(s)) to column {move.to_index + 1} "
            f"[onto {_card_label(_target_top(board, move))}]{empty}{reveal}"
        )
    return f"{index}. {move!r}"


def build_prompt(board: Board, valid_moves: list[Move]) -> list[dict]:
    """Return messages in OpenAI-compatible format (system + user)."""
    moves_text = "\n".join(
        _describe_move(board, m, i) for i, m in enumerate(valid_moves)
    )
    user_content = (
        f"Current board:\n{render(board)}\n\n"
        f"Available moves:\n{moves_text}\n\n"
        "Choose a move and explain your decision."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class InvalidPlannerResponse(Exception):
    pass


def parse_response(raw_text: str, valid_moves: list[Move]) -> tuple[int, str]:
    """
    Extract (move_index, reasoning_text) from the model's answer.
    Reasoning is all text BEFORE the json line; move_index is validated
    against the bounds of the valid moves list.
    """
    json_matches = list(re.finditer(r"\{[^{}]*\"move_index\"[^{}]*\}", raw_text))
    if not json_matches:
        raise InvalidPlannerResponse(
            f"No JSON with move_index found in the model's answer: {raw_text!r}"
        )
    last_match = json_matches[-1]
    try:
        data = json.loads(last_match.group(0))
        move_index = int(data["move_index"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise InvalidPlannerResponse(f"Broken JSON in the model's answer: {e}") from e

    # Soft fallback: if the model picked an out-of-range index (it happens with
    # small/weaker local models), clamp to the nearest valid move instead of
    # crashing the whole auto-play. Prefer "draw" (index 0) when available.
    if not (0 <= move_index < len(valid_moves)):
        if len(valid_moves) == 0:
            raise InvalidPlannerResponse("No valid moves passed to the planner.")
        move_index = 0
        reasoning = (
            "(fallback) model returned an out-of-range index; "
            "using the first valid move.\n" + raw_text[: last_match.start()].strip()
        )
        return move_index, reasoning

    reasoning = raw_text[: last_match.start()].strip()
    return move_index, reasoning
