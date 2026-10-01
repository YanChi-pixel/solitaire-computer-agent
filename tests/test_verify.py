"""Unit tests for the move-verification layer (pure geometry, no window)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from orchestrator.rule_engine import (
    Move,
)
from orchestrator.verify import check_shift, fan_step_for

STEP = 55  # real OPEN-card fan_step for card_h=176 (5/16 * 176)


def _ys(*values):
    return list(values)


def test_fan_step_for_ratio():
    assert fan_step_for(176) == 55  # 176 * 0.3125
    assert fan_step_for(100) == 31  # 100 * 0.3125 = 31.25 -> 31


def test_tableau_to_tableau_correct_shift():
    move = Move(kind="tableau_to_tableau", from_index=5, to_index=3, card_count=2)
    before = _ys(400, 400, 400, 400, 400, 500, 400)  # src=5 at 500, dst=3 at 400
    after = _ys(400, 400, 400, 400 + 2*STEP, 400, 500 - 2*STEP, 400)
    ok, reason = check_shift(before, after, move, STEP)
    assert ok, reason


def test_tableau_to_tableau_wrong_count_detected():
    # planned 2 cards, but only 1 actually moved (shift of 1*STEP, not 2*STEP)
    move = Move(kind="tableau_to_tableau", from_index=5, to_index=3, card_count=2)
    before = _ys(400, 400, 400, 400, 400, 500, 400)
    after = _ys(400, 400, 400, 400 + STEP, 400, 500 - STEP, 400)
    ok, reason = check_shift(before, after, move, STEP)
    assert not ok
    assert "source" in reason or "dest" in reason


def test_source_empties_onto_empty_dest():
    move = Move(kind="tableau_to_tableau", from_index=0, to_index=1, card_count=1)
    before = _ys(500, None, 400, 400, 400, 400, 400)
    after = _ys(None, 500, 400, 400, 400, 400, 400)  # src emptied, dst gained
    ok, reason = check_shift(before, after, move, STEP)
    assert ok, reason


def test_drag_missed_entirely_detected():
    move = Move(kind="tableau_to_tableau", from_index=5, to_index=3, card_count=1)
    before = _ys(400, 400, 400, 400, 400, 500, 400)
    after = list(before)  # nothing changed at all
    ok, reason = check_shift(before, after, move, STEP)
    assert not ok


def test_tableau_to_foundation_shift():
    move = Move(kind="tableau_to_foundation", from_index=2)
    before = _ys(400, 400, 500, 400, 400, 400, 400)
    after = _ys(400, 400, 500 - STEP, 400, 400, 400, 400)  # src rises by STEP
    ok, reason = check_shift(before, after, move, STEP)
    assert ok, reason


def test_draw_and_waste_to_foundation_have_no_fan_delta():
    d = Move(kind="draw")
    wf = Move(kind="waste_to_foundation")
    ys = _ys(400, 400, 400, 400, 400, 400, 400)
    assert check_shift(ys, ys, d, STEP)[0]
    assert check_shift(ys, ys, wf, STEP)[0]


def test_waste_to_tableau_gain():
    move = Move(kind="waste_to_tableau", to_index=4)
    before = _ys(400, 400, 400, 400, 500, 400, 400)
    after = _ys(400, 400, 400, 400, 500 + STEP, 400, 400)
    ok, reason = check_shift(before, after, move, STEP)
    assert ok, reason


def test_real_open_step_55_accepted_for_one_card():
    # Regression: the real OPEN-card step is 55px. A 1-card move shifts the
    # free card by 55px, which MUST be accepted — earlier 26px (0.15) and 18px
    # (0.102, the closed-back step) both made every move a false "not confirmed".
    move = Move(kind="tableau_to_tableau", from_index=4, to_index=2, card_count=1)
    before = _ys(400, 400, 400, 400, 500, 400, 400)
    after = _ys(400, 400, 400 + STEP, 400, 500 - STEP, 400, 400)
    ok, reason = check_shift(before, after, move, STEP)
    assert ok, reason


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
