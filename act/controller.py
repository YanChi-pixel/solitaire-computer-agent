"""
Act layer: the "hands" — real mouse control of the game window.

What it does:
- find the game window by its title (no hardcoded coordinates);
- capture a screenshot of exactly the window (mss, as tuned in session 1);
- click and drag cards via SendInput (the only method the UWP Solitaire
  app accepts — plain mouse_event is ignored).

This reuses proven code from session 1 (the "Solitaire" folder), but is
packaged as an importable module rather than ad-hoc scripts.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import numpy as np

try:
    import mss
    import win32gui
except ImportError:  # if pywin32/mss are missing
    win32gui = None
    mss = None


# ── 1. Low-level SendInput (click / drag) ──────────────────────────────
user32 = ctypes.windll.user32

ULONG_PTR = ctypes.c_size_t

class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT)]

class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

_MOVE = 0x0001
_LDOWN = 0x0002
_LUP = 0x0004
_ABS = 0x8000

# keyboard events (for Undo / Ctrl+Z and other hotkeys)
# KEYEVENTF_KEYUP = 0x0002 ; a plain key-down has flags=0.
_KEYEVENTF_KEYUP = 0x0002
_VK_CONTROL = 0x11
_VK_Z = 0x5A


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class _INPUTUNION_KEY(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT)]


class _INPUT_KEY(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION_KEY)]


def _send_key(vk: int, flags: int):
    """Send a single keyboard event via SendInput (the same mechanism as the
    mouse — the UWP Solitaire app ignores keybd_event, only SendInput works)."""
    i = _INPUT_KEY()
    i.type = 1  # INPUT_KEYBOARD
    i.u.ki = _KEYBDINPUT(vk, 0, flags, 0, 0)
    user32.SendInput(1, ctypes.byref(i), ctypes.sizeof(_INPUT_KEY))


def press_ctrl_z():
    """Undo the last move (Ctrl+Z) in Microsoft Solitaire Collection.

    Emits the chord via SendInput: Ctrl down, Z down/up, Ctrl up. The game
    accepts this as its "Undo" action; it reverts the game's own state to
    the previous confirmed position, so we don't have to reconstruct what a
    botched drag did on screen — the game rolls itself back.
    """
    _send_key(_VK_CONTROL, 0)
    time.sleep(0.05)
    _send_key(_VK_Z, 0)
    time.sleep(0.05)
    _send_key(_VK_Z, _KEYEVENTF_KEYUP)
    time.sleep(0.05)
    _send_key(_VK_CONTROL, _KEYEVENTF_KEYUP)
    time.sleep(0.3)


def undo_last_move(times: int = 1):
    """Roll back `times` moves via Ctrl+Z (each emits one undo)."""
    for _ in range(max(0, times)):
        press_ctrl_z()

# Offset of the game window's client area on the screen. Screenshot
# coordinates (0,0) correspond to the screen point (offset_x, offset_y).
# Updated every time we capture the window.
_offset_x = 0
_offset_y = 0


def _send(flags, dx=0, dy=0):
    i = _INPUT()
    i.type = 0
    i.u.mi = _MOUSEINPUT(dx, dy, 0, flags, 0, 0)
    user32.SendInput(1, ctypes.byref(i), ctypes.sizeof(_INPUT))


def _scr_to_abs(x, y):
    # Match the PROVEN working mapping from session 1: map against the
    # PRIMARY monitor size (GetSystemMetrics(0/1)), NOT the virtual screen.
    # The game runs on the primary monitor, and SendInput with ABSOLUTE
    # expects this mapping.
    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)
    ax = int(x * 65535 / (sw - 1))
    ay = int(y * 65535 / (sh - 1))
    return ax, ay


def _move(x, y):
    # Move via SendInput ABSOLUTE (the proven-working way), not SetCursorPos.
    ax, ay = _scr_to_abs(x, y)
    _send(_MOVE | _ABS, ax, ay)


def click_screen(x: int, y: int, double: bool = False):
    """Click at SCREENSHOT coordinates (x, y). Translated to screen coords via the window offset."""
    sx = int(x) + _offset_x
    sy = int(y) + _offset_y
    _move(sx, sy)
    time.sleep(0.1)
    _send(_LDOWN)
    time.sleep(0.06)
    _send(_LUP)
    time.sleep(0.1)
    if double:
        time.sleep(0.03)
        _send(_LDOWN)
        time.sleep(0.06)
        _send(_LUP)
        time.sleep(0.1)


def drag_screen(sx: int, sy: int, ex: int, ey: int, steps: int = 50):
    """
    Drag the mouse from (sx, sy) to (ex, ey).

    This is the EXACT configuration that works (mouse_ctrl.py, session 1):
    move + buttons ALL through SendInput (MOUSEEVENTF_ABSOLUTE for motion,
    LDOWN/LUP for buttons), with the primary-monitor mapping in _scr_to_abs.
    """
    # translate screenshot coords -> screen coords via the window offset
    sx = int(sx) + _offset_x
    sy = int(sy) + _offset_y
    ex = int(ex) + _offset_x
    ey = int(ey) + _offset_y
    _move(sx, sy)
    time.sleep(0.18)
    _send(_LDOWN)
    time.sleep(0.35)          # hold so the card "lifts"
    for i in range(1, steps + 1):
        x = sx + (ex - sx) * i / steps
        y = sy + (ey - sy) * i / steps
        _move(x, y)
        time.sleep(0.012)
    time.sleep(0.25)
    _send(_LUP)
    time.sleep(0.2)


# ── 2. Find the game window ────────────────────────────────────────────
def find_game_window(title_substring: str = "Solitaire & Casual Games") -> int:
    """
    Return the hwnd of the game window.

    We search by the EXACT game title ("Solitaire & Casual Games"), because
    the substring "Solitaire" also matches our own windows ("Solitaire AI -
    server", the browser). Return the first match with this exact name.
    """
    if win32gui is None:
        raise RuntimeError("pywin32 required: pip install pywin32 mss")
    result = []

    def callback(hwnd, _):
        text = win32gui.GetWindowText(hwnd)
        if (win32gui.IsWindowVisible(hwnd)
                and title_substring.lower() in text.lower()
                and "server" not in text.lower()
                and "bot" not in text.lower()
                and "chrome" not in text.lower()):
            result.append(hwnd)

    win32gui.EnumWindows(callback, None)
    if not result:
        raise RuntimeError(
            f"Game window '{title_substring}' not found. "
            "Is the game open and not minimized?"
        )
    return result[0]


def get_window_region(hwnd: int):
    """Return (left, top, width, height) of the window's client area in screen coords."""
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (left, top))
    right, bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    return left, top, right - left, bottom - top


# ── 3. Window screenshot ───────────────────────────────────────────────
def capture_window(hwnd: int) -> np.ndarray:
    """Capture the window area. Returns an RGB array (HxWx3).

    Also records the client-area offset into _offset_x/_offset_y, so that
    later click_screen/drag_screen (which take screenshot coords) map
    correctly to real screen coords even when the window has a border.
    """
    global _offset_x, _offset_y
    left, top, width, height = get_window_region(hwnd)
    # Свёрнутое/скрытое окно отдаёт GetClientRect = -32000,0,0 -> mss кидает
    # ScreenShotError (region zero/negative). Вместо мусорного исключения —
    # понятная ошибка, которую do_one_move уже умеет превращать в честный
    # "Table not visible" вместо бесконечного Stuck по устаревшей памяти.
    if width <= 0 or height <= 0 or left <= -32000 or top <= -32000:
        raise RuntimeError(
            f"Game window {hwnd} is minimized or hidden "
            f"(region {left},{top} {width}x{height}). Restore/focus it first."
        )
    _offset_x = left
    _offset_y = top
    with mss.mss() as sct:
        raw = sct.grab({"left": left, "top": top, "width": width, "height": height})
    return np.array(raw)[:, :, :3][:, :, ::-1]  # BGRA -> RGB


# ── 4. Window focus (so clicks go to the game, not elsewhere) ──────────
def focus_window(hwnd: int):
    """
    Gently bring the game to the foreground WITHOUT yanking focus away from
    every other window (which minimized the user's Cherry Studio / browser and
    caused "Stuck" because the mouse/eyes were still there).

    Order matters:
      1. If minimized, SW_RESTORE first (this alone un-minimizes without
         stealing foreground).
      2. Only then SetForegroundWindow — and only when the game is NOT already
         the foreground window. SetForegroundWindow on an ALREADY-foreground
         window is a no-op; calling it unconditionally every move was the
         "aggressive focus" that minimized everything else.

    UWP apps do need foreground to accept SendInput clicks, so we cannot drop
    SetForegroundWindow entirely — we just stop calling it when it is already
    focused (which is the common case once the game is up).
    """
    restored = False
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        restored = True

    fg = user32.GetForegroundWindow()
    if fg != hwnd or restored:
        # only re-foreground if we actually need to (minimized, or not focused)
        user32.SetForegroundWindow(hwnd)

    # give the window a beat to settle; short enough to not feel "grabby"
    time.sleep(0.2)
