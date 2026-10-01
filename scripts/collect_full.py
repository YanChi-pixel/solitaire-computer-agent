"""
Collect FULL face-up cards (not tiny corners) for retraining MobileNet.

The previous dataset used tiny 42x54 corners, which forces the classifier to
guess rank+face from a clipped symbol — this is why A/K/Q/J and suits were
confused. Retraining on the FULL card (top ~2/3 where rank + big suit symbol
are both visible) fixes this.

Run:  python scripts/collect_full.py
While playing, it saves full card crops to dataset_full/_unlabeled/.
"""

import sys
import time
import uuid
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import mss
import win32gui

from vision.screen_to_board import count_cards_in_column, find_tableau_columns

OUT_DIR = _ROOT / "dataset_full" / "_unlabeled"


def find_game_window(sub="Solitaire & Casual Games"):
    result = []
    def cb(hwnd, _):
        t = win32gui.GetWindowText(hwnd)
        if (win32gui.IsWindowVisible(hwnd) and sub.lower() in t.lower()
                and "server" not in t.lower() and "chrome" not in t.lower()):
            result.append(hwnd)
    win32gui.EnumWindows(cb, None)
    if not result:
        raise RuntimeError("Game window not found.")
    return result[0]


def capture_window(hwnd):
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (left, top))
    right, bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    with mss.mss() as sct:
        raw = sct.grab({"left": left, "top": top, "width": right-left, "height": bottom-top})
    return np.array(raw)[:, :, :3][:, :, ::-1]


def collect_once():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hwnd = find_game_window()
    img = capture_window(hwnd)
    H, W, _ = img.shape

    centers = find_tableau_columns(img)
    step = int(np.mean(np.diff(centers)))
    card_w = int(step * 0.78)  # measured: card ~131 at step 168
    card_h = int(card_w * 1.35)  # ~176

    top_y = int(H * 0.33)
    bottom_y = int(H * 0.65)

    saved = 0
    from PIL import Image

    def save_crop(crop):
        nonlocal saved
        if crop.shape[0] >= 50 and crop.shape[1] >= 50:
            fname = OUT_DIR / f"{uuid.uuid4().hex[:10]}.png"
            Image.fromarray(crop).save(fname)
            saved += 1

    # 1) 7 табличных колонок — ИГРАБЕЛЬНАЯ (нижняя, полностью видимая) карта.
    #    Именно она попадает в датасет чистой: перекрытые карты веера модель
    #    видит склеенными с соседями и учится на мусоре.
    for cx in centers:
        closed, oy, _ = count_cards_in_column(img, cx, top_y, bottom_y)
        if oy is None:
            continue
        left = cx - card_w // 2
        crop = img[oy:oy + card_h, left:left + card_w]
        save_crop(crop)

    # 2) верхняя карта выдачи (waste) — самый чистый одиночный источник.
    #    В верхней полосе y 100..300 ищем самую правую белую карту.
    r = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    b = img[:, :, 2].astype(int)
    white = (r > 200) & (g > 200) & (b > 200)
    top_zone = white[100:300, :]
    wcol = top_zone.sum(axis=0)
    segs = []
    start = None
    for x in range(W):
        if wcol[x] > 15 and start is None:
            start = x
        elif wcol[x] <= 15 and start is not None:
            if x - start > 15:
                segs.append((start, x - 1))
            start = None
    if start is not None:
        segs.append((start, W - 1))
    # берём самый правый широкий белый сегмент (это waste), пропуская UI справа
    for s, e in reversed(segs):
        if e - s > 60:  # похоже на карту (~131px), не на тонкую полоску
            waste_left = s
            waste_top = None
            # верх белой зоны в этом сегменте
            col_ys = np.where(top_zone[:, waste_left:waste_left + 60].any(axis=1))[0]
            if len(col_ys):
                waste_top = 100 + col_ys.min()
            if waste_top is not None:
                crop = img[waste_top:waste_top + card_h, waste_left:waste_left + card_w]
                save_crop(crop)
            break
    return saved


def loop(interval=3.0, max_captures=500):
    print(f"Collecting FULL cards. Play the game. Output: {OUT_DIR}")
    total = 0
    for i in range(max_captures):
        try:
            n = collect_once()
            total += n
            if n:
                print(f"[{i+1}] full cards: {n} (total {total})")
        except Exception as e:
            print(f"  skip: {e}")
        time.sleep(interval)
    print(f"Done. Total collected: {total}")


if __name__ == "__main__":
    loop()
