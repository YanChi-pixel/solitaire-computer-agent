"""Проверка vision-слоя на СОХРАНЁННОМ кадре против эталона, снятого глазами.

Эталон (прочитан с _live_frame.png вручную):
  [1] 9♠ 8♦ 7♠ 6♥                (0 рубашек, верх = 6♥)
  [2] ? 8♠                       (1 рубашка,  верх = 8♠)
  [3] ? 4♠                       (1 рубашка,  верх = 4♠)
  [4] ? ? ? K♦ Q♠ J♦ 10♣        (3 рубашки,  верх = 10♣)
  [5] ? ? ? Q♦                   (3 рубашки,  верх = Q♦)
  [6] ? ? 3♦                     (2 рубашки,  верх = 3♦)
  [7] ? ? ? ? ? ? 10♠            (6 рубашек,  верх = 10♠)
  waste = 4♣ K♥ 9♥ (верх 9♥); foundations ♥:A♥ ♦:A♦
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from orchestrator.rule_engine import RANK_NAMES
from vision import screen_to_board as s2b
from vision.recognizer import get_recognizer

frame = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_live_frame.png"
img = np.array(Image.open(frame).convert("RGB"))
H, W = img.shape[:2]
centers = s2b.find_tableau_columns(img)
card_w, card_h = s2b.measure_card_size(img, centers)
print(f"frame={frame.name} {W}x{H} centers={centers} card={card_w}x{card_h}")

rec = get_recognizer()
top_y, bottom_y = int(H * 0.33), int(H * 0.90)

TRUTH = [  # (closed, name)
    (0, "6♥"), (1, "8♠"), (1, "4♠"), (3, "10♣"), (3, "Q♦"), (2, "3♦"), (6, "10♠"),
]

ok_all = True
print("\nколонка | closed (ждём) | играбельная карта (ждём) | conf | низ стопки")
for i, cx in enumerate(centers):
    closed, open_y, n_open = s2b.count_cards_in_column(img, cx, top_y, bottom_y, card_w, card_h)
    sb = s2b.find_stack_bottom(img, cx, top_y, bottom_y, card_w)
    name, conf = "—", 0.0
    if open_y is not None:
        rank, suit, conf = rec.recognize_at_with_confidence(img, cx, open_y, card_w, card_h)
        name = f"{RANK_NAMES.get(rank, rank)}{suit.value}"
    exp_closed, exp_name = TRUTH[i]
    ok = (closed == exp_closed) and (name == exp_name)
    ok_all &= ok
    print(f"  [{i+1}]   | {closed} ({exp_closed})    | {name} ({exp_name})"
          f"{'' if ok else '  <<< MISMATCH'} | {conf:.3f} | {sb}")

board = s2b.screen_to_board(img, rec)
from server import render  # noqa: E402

print("\nдоска, как её видит CV:\n" + render(board))
print("\nИТОГ:", "ВСЁ СОВПАЛО С ЭТАЛОНОМ ✅" if ok_all else "ЕСТЬ РАСХОЖДЕНИЯ ❌")
