"""
DIAGNOSTIC: an overlapped card (only its top strip visible).

Demonstrates the core problem in our Klondike solitaire AI:

  - MobileNet-v2 (finetuned) reads FULLY-VISIBLE cards with conf ~1.0.
  - But the BOTTOM (playable) card inside a multi-card column fan is
    PARTIALLY OVERLAPPED by the card above it, so only its top strip
    (rank + suit corner) is visible. The model, trained on FULL cards
    (131x176), reads that overlapped card unreliably (conf 0.3..0.8),
    and sometimes emits a WRONG card with spuriously HIGH confidence
    when the crop slips down into the felt/shadow.

This script:
  1. loads a saved screenshot snapshot (or captures the live window),
  2. for the last few columns, walks DOWN the column in ~8px steps,
     crops a full-card window at each step and prints (rank, suit, conf),
  3. makes the confusion visible: a column where the top open card reads
     clearly, but scanning further down flips through wrong cards.

Run it against _snapshot.npy (or with the live game if you have the window).

The open question this script illustrates:
  A) a crop/geometry bug that can be fixed in code (read the bottom card's
     top edge correctly), or
  B) a data limitation — the model would also need training on
     "truncated / overlapped card tops" to read buried cards reliably.
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

    # --- load an image: prefer saved snapshot, else capture live window ---
    snap = _ROOT / "scripts" / "_snapshot.npy"
    if snap.exists():
        img = np.load(snap)
        print(f"[using saved snapshot {snap}  shape={img.shape}]")
    else:
        hwnd = act.find_game_window()
        act.focus_window(hwnd)
        img = act.capture_window(hwnd)
        print(f"[captured live window  shape={img.shape}]")

    centers = find_tableau_columns(img)
    cw, ch = measure_card_size(img, centers)
    print(f"columns={centers}  card={cw}x{ch}\n")

    # focus on the last 3 columns (rightmost, usually deepest fans)
    for ci in [4, 5, 6]:
        cx = centers[ci]
        print(f"=== column[{ci}] cx={cx}  (y-scan, step 8px) ===")
        for top in range(380, 600, 8):
            crop = img[top:top + ch, cx - cw // 2:cx + cw // 2]
            rank, suit, conf = rec.recognize_with_confidence(crop)
            bar = "#" * int(conf * 20)
            print(f"  top={top:4d}: {rank:>2}{suit.value}  conf={conf:.3f} {bar}")
        print()


if __name__ == "__main__":
    main()
