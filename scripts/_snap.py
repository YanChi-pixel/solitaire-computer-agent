"""Snapshot: memory vs real screen side by side."""
import json
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from act import controller as act
from orchestrator.rule_engine import render
from vision.recognizer import get_recognizer
from vision.screen_to_board import screen_to_board

mem = json.loads(urllib.request.urlopen('http://127.0.0.1:8000/dump').read())
print("=== MEMORY ===")
print(mem["board"])

rec = get_recognizer()
hwnd = act.find_game_window()
act.focus_window(hwnd)
img = act.capture_window(hwnd)
print(f"\n=== SCREEN {img.shape} ===")
print(render(screen_to_board(img, recognizer=rec)))
