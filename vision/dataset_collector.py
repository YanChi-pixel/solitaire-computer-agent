"""
Dataset collector — runs on YOUR computer (not in a sandbox), because it
needs a real screen with Microsoft Solitaire Collection open.

What it does:
1. Finds the game window by title ("Solitaire" in the name).
2. Captures a screenshot of exactly the window (via its bounds), not the
   whole screen — that's the position/scale robustness discussed earlier.
3. Calibrates geometry and crops the corners of face-up cards.
4. Saves each corner to dataset/_unlabeled/ with a unique name —
   later they are labeled by label_corners.py.

Requires (on Windows): pip install pywin32 mss --break-system-packages
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import numpy as np

# Allows running as `python vision/dataset_collector.py` from any cwd by
# adding the project root (sol-dev/) to sys.path for absolute `vision.*`
# imports. Without it, sys.path[0] = vision/ and the package won't resolve.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import mss
    import win32gui
except ImportError:
    win32gui = None
    mss = None

from vision.calibration import calibrate
from vision.card_extraction import extract_visible_corners

TABLE_BG = (24, 123, 78)  # tuned to real screenshots; change for another theme
UNLABELED_DIR = _PROJECT_ROOT / "dataset" / "_unlabeled"


def find_game_window(title_substring: str = "Solitaire"):
    """Return the hwnd of a window whose title contains title_substring."""
    result = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and title_substring.lower() in win32gui.GetWindowText(hwnd).lower():
            result.append(hwnd)

    win32gui.EnumWindows(callback, None)
    if not result:
        raise RuntimeError(
            f"Window with '{title_substring}' in the title not found. "
            "Is the game open and not minimized?"
        )
    return result[0]


def capture_window(hwnd) -> np.ndarray:
    """Capture exactly the window area (not the whole screen)."""
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (left, top))
    right, bottom = win32gui.ClientToScreen(hwnd, (right, bottom))

    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        raw = sct.grab(monitor)
        return np.array(raw)[:, :, :3][:, :, ::-1]  # BGRA -> RGB


def collect_once(save_dir: Path = UNLABELED_DIR) -> int:
    """One cycle: screenshot -> calibrate -> crop corners -> save. Returns count of saved corners."""
    save_dir.mkdir(parents=True, exist_ok=True)

    hwnd = find_game_window()
    image = capture_window(hwnd)

    geometry = calibrate(image, table_bg_rgb=TABLE_BG, expected_columns=7)
    corners = extract_visible_corners(image, geometry, TABLE_BG)

    from PIL import Image as PILImage
    saved = 0
    for c in corners:
        fname = save_dir / f"{uuid.uuid4().hex[:10]}_col{c.column_index}.png"
        PILImage.fromarray(c.crop).save(fname)
        saved += 1
    return saved


def collect_loop(interval_seconds: float = 3.0, max_captures: int = 100):
    """
    Periodically crops corners while you play by hand (or via Solver moves).
    Each new layout/move reveals new face-up cards — so one game can yield
    dozens of distinct examples with little effort.
    """
    print(f"Collecting. Play the game — every {interval_seconds}s I'll crop face-up cards.")
    print("Stop: Ctrl+C")
    total = 0
    for i in range(max_captures):
        try:
            saved = collect_once()
            total += saved
            print(f"[{i+1}/{max_captures}] corners saved: {saved} (total: {total})")
        except Exception as e:
            print(f"  skip (error: {e})")
        time.sleep(interval_seconds)
    print(f"Done. Total collected: {total} in {UNLABELED_DIR}")


if __name__ == "__main__":
    if win32gui is None:
        print("Requires pywin32 and mss: pip install pywin32 mss --break-system-packages")
    else:
        collect_loop()
