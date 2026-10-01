"""
Move verification layer (architecture B, the missing piece).

``apply_move`` mutates the in-memory Board deterministically; ``act`` performs
a real mouse drag that is NOT guaranteed to land. The recurring bug ("planned to
move 2 cards, only 1 went", "ghost cards in memory", "shuttling the same stack
back and forth") is exactly the absence of the check BETWEEN them: memory was
advanced without confirming the physical drag changed the screen as intended.

This module closes that gap. After a move is performed, before planning the
next, we re-read ONLY what is physically guaranteed to be fully visible (the
free card of each pile, conf ~1.0) and compare it against memory. Buried
(overlapped) cards are never re-recognized.

The core signal is the VERTICAL SHIFT of a column's free card (``open_y``).
A Klondike column is a fan with a fixed vertical step ``fan_step``. When N open
cards leave a column, the free card rises by ``N * fan_step``; when N cards
land, the target's free card sinks by ``N * fan_step``. This delta is purely
geometric, needs no ML, and is independent of absolute positions — it directly
catches the "the drag moved a different number of cards than planned" bug.

On failure we DO NOT repair memory toward the drifted screen (that built the
ghost cards). The caller rolls the SCREEN back via Undo and retries, then
re-applies the move to memory only after it is confirmed.
"""

from __future__ import annotations

from orchestrator.rule_engine import (
    UNKNOWN_RANK,
    Board,
    Card,
)


def fan_step_for(card_h: int) -> int:
    """
    Vertical step of the OPEN-card fan (the value that governs how far a free
    card shifts when one open card is added/removed).

    Measured empirically on Microsoft Solitaire Collection: an open card step
    is exactly 5/16 of the card height (= 55px at card_h=176), NOT the closed
    back step (which is ~0.102 * card_h = 18px and is unrelated to open-card
    stacking). The earlier 0.15 (26px) and 0.102 (18px) values were both wrong
    and caused every move to fail verification.
    """
    return max(8, int(round(card_h * 0.3125)))


def _top_open_card(board: Board, col_idx: int):
    col = board.tableau[col_idx] if col_idx < len(board.tableau) else []
    for c in reversed(col):
        if c.face_up:
            return c
    return None


def _cards_match(a: Card | None, b: Card | None) -> bool:
    """Compare two fully-visible cards.

    A card whose identity is UNKNOWN (rank 0, e.g. a just-flipped card that
    ``apply_move`` turned face-up but left unidentified) matches anything —
    its identity is filled in later by ``_sync_revealed_card``. Only a
    KNOWN-vs-KNOWN mismatch is a failure; this prevents a freshly-revealed
    card from causing a false "not confirmed".
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a.rank == UNKNOWN_RANK or b.rank == UNKNOWN_RANK:
        return True  # one side is unidentified -> not a provable mismatch
    return a.rank == b.rank and a.suit == b.suit


# ── pure geometry: the shift check is testable without any image ─────────
def check_shift(
    ys_before: list,
    ys_after: list,
    move,
    step: int,
    tolerance: int = 6,
) -> tuple[bool, str]:
    """
    Pure function: given the free-card open_y of every column before and after
    a move, decide whether the shift matches what the move declares.

    ``ys_before``/``ys_after`` are length-7 lists of int|None. ``step`` is the
    fan step in pixels. Returns (ok, reason) — no image, no side effects.
    """
    if move.kind == "tableau_to_tableau":
        src, dst, n = move.from_index, move.to_index, move.card_count
        exp_src = -n * step
        exp_dst = n * step

        sb, sa = ys_before[src], ys_after[src]
        db, da = ys_before[dst], ys_after[dst]

        # source: the free card rises by n*step, OR the column empties
        if sb is None:
            return False, f"source col {src + 1} had no free card before the move"
        if sa is None:
            # column emptied — verify dest gained the full pile (covered below)
            pass
        elif abs((sa - sb) - exp_src) > tolerance:
            return False, (f"source col {src + 1} shifted {(sa - sb)}px, "
                           f"expected {exp_src}px (moved {n} cards)")

        # destination: free card sinks by n*step, or a card appears on empty
        if db is None:
            if da is None:
                return False, "dest column stayed empty — drag did not land"
            # was empty, now has cards: we can't measure a delta, accept
        elif da is None:
            return False, "dest column lost its free card — unexpected"
        elif abs((da - db) - exp_dst) > tolerance:
            return False, (f"dest col {dst + 1} shifted {(da - db)}px, "
                           f"expected {exp_dst}px (received {n} cards)")

    elif move.kind == "tableau_to_foundation":
        src, n = move.from_index, 1
        sb, sa = ys_before[src], ys_after[src]
        if sb is None:
            return False, f"source col {src + 1} had no free card"
        if sa is not None and abs((sa - sb) + step) > tolerance:
            return False, f"source col {src + 1} did not lose one card"

    elif move.kind == "waste_to_tableau":
        dst, n = move.to_index, 1
        db, da = ys_before[dst], ys_after[dst]
        if db is None:
            if da is None:
                return False, "dest col stayed empty — waste card did not land"
        elif da is None:
            return False, "dest col lost its free card — unexpected"
        elif abs((da - db) - step) > tolerance:
            return False, f"dest col {dst + 1} did not gain one card"

    # draw / waste_to_foundation: no tableau fan delta
    return True, "ok"


def column_open_ys(image, centers, table_top_y, bottom_y, card_w, card_h):
    """Free-card top y (open_y) per column, via the proven count_cards_in_column."""
    from vision.screen_to_board import count_cards_in_column
    open_ys = []
    for cx in centers:
        closed, open_y, _ = count_cards_in_column(
            image, cx, table_top_y, bottom_y, card_w, card_h
        )
        open_ys.append(open_y)
    return open_ys


def verify_quantity(image_before, image_after, positions, move, fan_step=None, tolerance=6):
    """
    Geometry check: did the drag move exactly N cards between src and dst?
    Extracts open_y before/after and delegates to the pure ``check_shift``.

    ``fan_step`` is the OPEN-card vertical step (default: ``fan_step_for``,
    i.e. 5/16 * card_h). It must match the physical open-card fan, NOT the
    closed-back step.
    """
    centers = positions["centers"]
    card_w = positions["card_w"]
    card_h = positions["card_h"]
    table_top_y = positions["table_top_y"]
    bottom_y = positions["bottom_y"]

    if fan_step is None:
        fan_step = fan_step_for(card_h)

    ys_before = column_open_ys(image_before, centers, table_top_y, bottom_y, card_w, card_h)
    ys_after = column_open_ys(image_after, centers, table_top_y, bottom_y, card_w, card_h)
    return check_shift(ys_before, ys_after, move, fan_step, tolerance)


def resolve_unknown(card: Card | None, image_after, cx: int, open_y, card_w: int, card_h: int, recognize) -> Card | None:
    """
    If `card` is a freshly-revealed placeholder (UNKNOWN_RANK), recognize it
    NOW (it's guaranteed fully visible at this exact moment) and return a
    NEW Card with the real identity. If it's already known, return it as-is.
    This is the missing "write", not just "compare" — without this, every
    flipped/drawn card stays UNKNOWN forever and can never be used to
    generate real moves or be verified again.
    """
    if card is None or card.rank != UNKNOWN_RANK or open_y is None:
        return card
    rank, suit, conf = recognize(image_after, cx, open_y, card_w, card_h)
    return Card(rank, suit, face_up=True)


def verify_identity(image_after, mem_after, positions, recognize, move):
    """
    Confirm the ONE newly-visible card at an exactly known region, AND
    resolve it into `mem_after` if it was still an UNKNOWN placeholder
    (a just-flipped or just-drawn card). This mutates mem_after in place —
    the caller keeps using the same board object afterwards.
    """
    centers = positions["centers"]
    card_w = positions["card_w"]
    card_h = positions["card_h"]
    table_top_y = positions["table_top_y"]
    bottom_y = positions["bottom_y"]

    from vision.screen_to_board import count_cards_in_column

    checked_kinds = (
        "tableau_to_tableau", "tableau_to_foundation",
        "waste_to_tableau", "waste_to_foundation",
    )
    if move.kind not in checked_kinds:
        return True, "n/a"

    # --- source column (tableau moves only; waste has no "source column") ---
    if move.kind in ("tableau_to_tableau", "tableau_to_foundation"):
        src = move.from_index
        cx = centers[src]
        closed, open_y, _ = count_cards_in_column(image_after, cx, table_top_y, bottom_y, card_w, card_h)
        mem_top = _top_open_card(mem_after, src)
        if open_y is not None and mem_top is not None:
            if mem_top.rank == UNKNOWN_RANK:
                # just-revealed card: resolve it INTO memory, don't just compare
                resolved = resolve_unknown(mem_top, image_after, cx, open_y, card_w, card_h, recognize)
                mem_after.tableau[src][-1] = resolved
            else:
                rank, suit, conf = recognize(image_after, cx, open_y, card_w, card_h)
                if rank != mem_top.rank or suit != mem_top.suit:
                    return False, (f"source col {src + 1} revealed card: "
                                   f"screen={rank}{suit.value} mem={mem_top}")

    # --- destination column (tableau_to_tableau only) ---
    if move.kind == "tableau_to_tableau":
        dst = move.to_index
        cx = centers[dst]
        closed, open_y, _ = count_cards_in_column(image_after, cx, table_top_y, bottom_y, card_w, card_h)
        mem_top = _top_open_card(mem_after, dst)
        if open_y is not None and mem_top is not None and mem_top.rank != UNKNOWN_RANK:
            rank, suit, conf = recognize(image_after, cx, open_y, card_w, card_h)
            if rank != mem_top.rank or suit != mem_top.suit:
                return False, (f"dest col {dst + 1} bottom card: "
                               f"screen={rank}{suit.value} mem={mem_top}")

    # --- waste (draw / waste_to_tableau / waste_to_foundation) ---
    # ONLY the rightmost (topmost, playable) waste card is ever checked —
    # the same anchor Layout.waste_xy already uses. Cards to its left in the
    # fan are buried and are NEVER fed to the recognizer, by the same
    # principle as tableau: we already know them, we placed them there.
    if move.kind in ("waste_to_tableau", "waste_to_foundation") and mem_after.waste:
        # after popping the played card, the NEW top of waste (if any) is
        # the card drawn just before it — already known, nothing to resolve.
        # Nothing to verify here: waste shrank, no new card became visible.
        pass

    return True, "ok"


def resolve_new_waste_card(mem_after, image_after, waste_cx, waste_top_y, card_w, card_h, recognize):
    """
    Call this right after a `draw` move (stock -> waste) is confirmed on
    screen — the freshly drawn card is, at that instant, guaranteed to be
    the fully-visible rightmost waste card. Resolves it and WRITES it into
    mem_after.waste[-1], replacing the UNKNOWN placeholder apply_move put
    there. Without this call, waste identities never resolve and every
    waste-based move stays permanently unverifiable.
    """
    if not mem_after.waste or mem_after.waste[-1].rank != UNKNOWN_RANK:
        return mem_after
    resolved = resolve_unknown(
        mem_after.waste[-1], image_after, waste_cx, waste_top_y, card_w, card_h, recognize
    )
    mem_after.waste[-1] = resolved
    return mem_after


def verify_board_tops(image_after, mem_after, recognize=None):
    """
    Broadest anchor check: every column's free card + waste + foundations match
    memory (only fully-visible cards, via screen_to_board).

    ⚠️ CRITICAL: the `fresh` board built here is DISPOSABLE — it exists only
    to compare against memory and is thrown away immediately after. It must
    NEVER be adopted as the new `mem_after` (or any part of it, e.g.
    `fresh.waste`) on failure. `screen_to_board()` always returns waste as a
    length-1 list and tableau columns without their buried history — using
    it as a replacement for memory is not a smaller bug, it IS the bug that
    causes buried cards and waste history to be silently deleted. On
    failure here, the caller's ONLY valid response is: Undo the physical
    screen back to the last confirmed state and retry the SAME move against
    the UNCHANGED existing memory — never re-derive memory from this read.
    """
    from vision.screen_to_board import screen_to_board
    try:
        fresh = screen_to_board(image_after, recognizer=recognize)
    except Exception as e:
        return False, f"screen read failed: {e}"

    for ci in range(7):
        if not _cards_match(_top_open_card(mem_after, ci), _top_open_card(fresh, ci)):
            return False, (f"col {ci + 1} top: mem={_top_open_card(mem_after, ci)} "
                           f"screen={_top_open_card(fresh, ci)}")

    mem_w = mem_after.waste[-1] if mem_after.waste else None
    ref_w = fresh.waste[-1] if fresh.waste else None
    if not _cards_match(mem_w, ref_w):
        return False, f"waste: mem={mem_w} screen={ref_w}"

    for s in mem_after.foundation:
        mem_f = mem_after.foundation[s][-1] if mem_after.foundation[s] else None
        ref_f = fresh.foundation.get(s, [])
        ref_top = ref_f[-1] if ref_f else None
        if not _cards_match(mem_f, ref_top):
            return False, f"foundation {s.value}: mem={mem_f} screen={ref_top}"

    return True, "ok"
