"""
DIAGNOSTIC: how reliably does the card recognizer read
an OPEN card vs a PARTIALLY-OVERLAPPED card inside a column fan (stack).

A Klondike column contains a fan of open cards, top->bottom by rank, e.g.
  5h (top), 4c, 3d (bottom = "playable").
Only the bottom card is fully visible; the ones above it are partially
covered by the card below, so only their top strip (rank + suit corner)
shows. The model (MobileNet, trained on FULL cards) reads the fully-visible
bottom card with conf~1.0, but the overlapped cards erratically (conf~0.3).

This script scans a column top-to-bottom in small steps and prints
(rank, suit, confidence) for each crop, making the drop-off visible.

The core question this exposes (for the human):
  To move a STACK of several cards, we must know every card in the fan.
  But buried cards can't be read reliably from pixels. Should we (1) train
  the model on "truncated card tops", (2) never move stacks, or (3) read a
  card only at the moment it becomes fully visible (lazy), keeping the rest
  in memory?

Run on the developer machine with the solitaire window open, or against a
saved snapshot _snapshot.npy.
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from act import controller as act
from vision.recognizer import get_recognizer
from vision.screen_to_board import find_tableau_columns, measure_card_size


def main():
    rec = get_recognizer()

    snap = _ROOT / "scripts" / "_snapshot.npy"
    if snap.exists():
        img = np.load(snap)
        print(f"[using snapshot {snap}  shape={img.shape}]")
    else:
        hwnd = act.find_game_window()
        act.focus_window(hwnd)
        img = act.capture_window(hwnd)
        print(f"[live window  shape={img.shape}]")

    centers = find_tableau_columns(img)
    cw, ch = measure_card_size(img, centers)
    print(f"columns={centers}  card={cw}x{ch}\n")

    # scan the LAST two columns (deepest fans, most overlapped cards)
    for ci in [5, 6]:
        cx = centers[ci]
        print(f"=== column[{ci + 1}] cx={cx}  (top->bottom scan, step 8px) ===")
        for top in range(400, 580, 8):
            crop = img[top:top + ch, cx - cw // 2:cx + cw // 2]
            rank, suit, conf = rec.recognize_with_confidence(crop)
            bar = "#" * int(conf * 20)
            print(f"  y={top:4d}: {rank:>2}{suit.value}  conf={conf:.3f} {bar}")
        print()


if __name__ == "__main__":
    main()
