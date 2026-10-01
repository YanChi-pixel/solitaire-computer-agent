"""
Rule Engine for Klondike solitaire (standard "draw 1" variant).

This module has a single job: turn the board state into a list of valid
moves and apply the chosen move. No CV, no LLM. Pure deterministic logic
that is easy to cover with unit tests.

State is represented in "logical" coordinates (column index, stack
position), not pixels — translation to screen coordinates for SendInput
happens at the Act layer, not here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"
    UNKNOWN = "?"   # for a face-down card whose suit we don't know yet

    @property
    def is_red(self) -> bool:
        return self in (Suit.HEARTS, Suit.DIAMONDS)


RANK_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}

# Rank of a face-down (unknown) card. We use 0 — no such rank exists in a
# real deck (1..13), so it unambiguously means "unknown, face-down".
UNKNOWN_RANK = 0

# The game is "draw-3": one click on the stock reveals up to 3 cards in a
# waste fan (fewer if the stock holds less). Only the TOP (rightmost) card of
# the fan is fully visible and playable; the ones below it are only partially
# shown, so their identity stays UNKNOWN until they reach the top.
DRAW_COUNT = 3

# Максимальный размер стопки, переносимой одним tableau→tableau ходом.
# Выше 4 карт drag становится физически ненадёжным в UWP-версии игры
# (жест срывается, верификация отклоняет ход), а выигрыша такой большой
# перенос не приносит — проще собрать/разобрать стопку по частям.
MAX_RUN_MOVE = 4


@dataclass(frozen=True)
class Card:
    rank: int  # 1..13, where 1 = Ace
    suit: Suit
    face_up: bool = False

    def flipped(self, face_up: bool) -> "Card":
        return Card(self.rank, self.suit, face_up)

    def __repr__(self) -> str:
        if self.rank == UNKNOWN_RANK:
            # face-down card — we know neither rank nor suit
            return "*" if self.face_up else "?"
        r = RANK_NAMES.get(self.rank, str(self.rank))
        face = "" if self.face_up else "*"
        return f"{r}{self.suit.value}{face}"


REAL_SUITS = (Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES)


def full_deck() -> list[Card]:
    return [Card(rank, suit) for suit in REAL_SUITS for rank in range(1, 14)]


class Location(Enum):
    TABLEAU = "tableau"     # 7 columns, index 0..6
    FOUNDATION = "foundation"  # 4 piles, one per suit
    STOCK = "stock"          # draw pile
    WASTE = "waste"          # face-up card on top of the draw pile


@dataclass
class Board:
    tableau: list[list[Card]] = field(default_factory=lambda: [[] for _ in range(7)])
    foundation: dict[Suit, list[Card]] = field(
        default_factory=lambda: {s: [] for s in REAL_SUITS}
    )
    stock: list[Card] = field(default_factory=list)
    waste: list[Card] = field(default_factory=list)

    @staticmethod
    def new_game(seed: Optional[int] = None) -> "Board":
        rng = random.Random(seed)
        deck = full_deck()
        rng.shuffle(deck)
        board = Board()
        idx = 0
        for col in range(7):
            for row in range(col + 1):
                is_last = row == col
                card = deck[idx].flipped(is_last)
                board.tableau[col].append(card)
                idx += 1
        board.stock = [c.flipped(False) for c in deck[idx:]]
        return board

    def clone(self) -> "Board":
        return Board(
            tableau=[list(col) for col in self.tableau],
            foundation={s: list(cards) for s, cards in self.foundation.items()},
            stock=list(self.stock),
            waste=list(self.waste),
        )

    def is_won(self) -> bool:
        return all(len(cards) == 13 for cards in self.foundation.values())


@dataclass(frozen=True)
class Move:
    """A move described in logical coordinates (no pixels)."""

    kind: str  # "tableau_to_tableau" | "tableau_to_foundation" |
               # "waste_to_tableau" | "waste_to_foundation" | "draw"
    from_loc: Optional[Location] = None
    from_index: Optional[int] = None   # column/suit index, if applicable
    card_count: int = 1                # how many cards to move (tableau->tableau)
    to_loc: Optional[Location] = None
    to_index: Optional[int] = None

    def __repr__(self) -> str:
        # human-readable description (for logs / web UI). Column indices are
        # shown 1-based for humans (internal indices stay 0-based). Draw has no
        # from/to, so a raw repr looked broken — this reads normally instead.
        if self.kind == "draw":
            return "Draw a card (stock → waste)"
        if self.kind == "waste_to_foundation":
            return "Move waste card to foundation"
        if self.kind == "waste_to_tableau":
            return f"Move waste card to column {self.to_index + 1}"
        if self.kind == "tableau_to_foundation":
            return f"Move top of column {self.from_index + 1} to foundation"
        if self.kind == "tableau_to_tableau":
            return (
                f"Move {self.card_count} card(s) from column "
                f"{self.from_index + 1} to column {self.to_index + 1}"
            )
        return self.kind


def _can_stack_tableau(moving: Card, target: Optional[Card]) -> bool:
    """Can card `moving` be placed on `target` in a tableau column."""
    if target is None:
        return moving.rank == 13  # only a King on an empty column
    return target.face_up and target.rank == moving.rank + 1 and target.suit.is_red != moving.suit.is_red


def _can_stack_foundation(moving: Card, foundation_pile: list[Card]) -> bool:
    if not foundation_pile:
        return moving.rank == 1  # only an Ace on an empty foundation
    top = foundation_pile[-1]
    return top.suit == moving.suit and top.rank + 1 == moving.rank


def get_valid_moves(board: Board) -> list[Move]:
    moves: list[Move] = []

    # 1. Draw from stock (or flip waste back into stock if stock is empty)
    if board.stock or board.waste:
        moves.append(Move(kind="draw"))

    # 2. Waste -> foundation / tableau
    if board.waste:
        top = board.waste[-1]
        if top.suit != Suit.UNKNOWN:
            if _can_stack_foundation(top, board.foundation[top.suit]):
                moves.append(
                    Move(kind="waste_to_foundation", from_loc=Location.WASTE,
                         to_loc=Location.FOUNDATION, to_index=top.suit)
                )
            for col_idx, col in enumerate(board.tableau):
                target = col[-1] if col else None
                if _can_stack_tableau(top, target):
                    moves.append(
                        Move(kind="waste_to_tableau", from_loc=Location.WASTE,
                             to_loc=Location.TABLEAU, to_index=col_idx)
                    )

    # 3. Tableau -> foundation (only the top face-up card of a column)
    for col_idx, col in enumerate(board.tableau):
        if not col or not col[-1].face_up:
            continue
        top = col[-1]
        if top.suit == Suit.UNKNOWN:
            continue
        if _can_stack_foundation(top, board.foundation[top.suit]):
            moves.append(
                Move(kind="tableau_to_foundation", from_loc=Location.TABLEAU,
                     from_index=col_idx, to_loc=Location.FOUNDATION,
                     to_index=top.suit)
            )

    # 4. Tableau -> tableau (move one card or a valid subsequence)
    for src_idx, src_col in enumerate(board.tableau):
        # find the longest valid (alternating color, descending rank)
        # face-up run at the END of the column. We scan the column BOTTOM-up;
        # `card` (current, higher up) must sit on `face_up_run[0]` (the card
        # just below it), so: rank(card) == rank(run_top) + 1 and colors differ.
        face_up_run = []
        for card in reversed(src_col):
            if not card.face_up:
                break
            if face_up_run and not (
                card.suit.is_red != face_up_run[0].suit.is_red
                and card.rank == face_up_run[0].rank + 1
            ):
                break
            face_up_run.insert(0, card)

        for run_len in range(1, len(face_up_run) + 1):
            # Ограничение размера переносимой стопки. Длинный drag (7+ карт)
            # в Microsoft Solitaire Collection физически НЕНАДЁЖЕН — жест
            # срывается, стопка «не долетает», и верификация по fan_step
            # отклоняет ход (реальный баг: 7 карт → сдвиг 285px из ожидаемых
            # 385px → 3 провала → автоигра вставала). Переносить больше 4 карт
            # почти никогда не нужно для выигрыша — эффективнее двигать по
            # 1-3 карты. Лишние карты в вершине стопки всегда можно перенести
            # потом отдельными ходами.
            if run_len > MAX_RUN_MOVE:
                break
            moving_card = face_up_run[-run_len]  # bottom card of the moved block
            for dst_idx, dst_col in enumerate(board.tableau):
                if dst_idx == src_idx:
                    continue
                target = dst_col[-1] if dst_col else None
                if _can_stack_tableau(moving_card, target):
                    moves.append(
                        Move(kind="tableau_to_tableau", from_loc=Location.TABLEAU,
                             from_index=src_idx, card_count=run_len,
                             to_loc=Location.TABLEAU, to_index=dst_idx)
                    )

    return moves


def apply_move(board: Board, move: Move) -> Board:
    """Return a NEW board state after the move (board is not mutated)."""
    b = board.clone()

    if move.kind == "draw":
        if b.stock:
            # ВАЖНО (после разбора реального move_log.log): "замена" здесь
            # была неверна и объективно теряла карты — лог показывает это
            # трижды до финального краха (9♣ и 10♦ выброшены при следующих
            # draw, ещё не будучи сыгранными). Реальная физика draw-3: карты
            # НИКОГДА не исчезают из waste, кроме как через foundation/
            # tableau или полный recycle. Непройденный остаток просто
            # перестаёт быть "верхним" (играбельным) — но остаётся в куче и
            # правильно всплывает позже. Это ровно append, не замена.
            #
            # Мы НЕ пытаемся предсказать identity новых карт — они уходят
            # как UNKNOWN-плейсхолдеры (или сохраняют identity, если она уже
            # была в b.stock — так ведёт себя чистый симулятор). Реальная
            # identity верхней карты каждый раз читается лениво с экрана
            # (server.py::_sync_revealed_card) — вопрос "какая карта стала
            # доступна после снятия верхней" решается не предсказанием
            # состава окна, а простым повторным взглядом на экран, и это
            # работает одинаково для любого источника всплывшей карты.
            n = min(DRAW_COUNT, len(b.stock))
            b.waste.extend(b.stock.pop().flipped(True) for _ in range(n))
        else:
            # recycle: перевернуть ВЕСЬ waste обратно в stock, с сохранением
            # порядка вытягивания — корректно ТОЛЬКО если waste и правда
            # хранит полную историю (append), а не последнее окно.
            b.stock = [c.flipped(False) for c in reversed(b.waste)]
            b.waste = []

    elif move.kind == "waste_to_foundation":
        card = b.waste.pop()
        b.foundation[card.suit].append(card)

    elif move.kind == "waste_to_tableau":
        card = b.waste.pop()
        b.tableau[move.to_index].append(card)

    elif move.kind == "tableau_to_foundation":
        card = b.tableau[move.from_index].pop()
        b.foundation[card.suit].append(card)
        _flip_new_top(b, move.from_index)

    elif move.kind == "tableau_to_tableau":
        src = b.tableau[move.from_index]
        moving = src[-move.card_count:]
        del src[-move.card_count:]
        b.tableau[move.to_index].extend(moving)
        _flip_new_top(b, move.from_index)

    else:
        raise ValueError(f"Unknown move kind: {move.kind}")

    return b


def _flip_new_top(board: Board, col_idx: int) -> None:
    """After removing a card, flip the new top card of the column if it was face-down."""
    col = board.tableau[col_idx]
    if col and not col[-1].face_up:
        col[-1] = col[-1].flipped(True)


def render(board: Board) -> str:
    """Text representation of the board — for logs and debugging."""
    lines = []
    stock_str = f"Stock({len(board.stock)})"
    waste_str = f"Waste: {board.waste[-1] if board.waste else '-'}"
    found_str = " ".join(
        f"{s.value}:{f[-1] if f else '-'}" for s, f in board.foundation.items()
    )
    lines.append(f"{stock_str}   {waste_str}   Foundations: {found_str}")
    lines.append("-" * 60)
    for i, col in enumerate(board.tableau):
        # columns are shown 1-based for human readability. An EMPTY column is
        # marked explicitly so the LLM understands it can receive a King.
        if col:
            lines.append(f"[{i + 1}] " + " ".join(repr(c) for c in col))
        else:
            lines.append(f"[{i + 1}] (empty)")
    return "\n".join(lines)
