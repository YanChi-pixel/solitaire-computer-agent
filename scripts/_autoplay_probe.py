"""Живой прогон: N ходов через API + НЕЗАВИСИМАЯ проверка «память vs экран».

После каждого хода:
  1. POST /move  (сервер: LLM -> drag -> verify -> память)
  2. GET  /dump  (память сервера)
  3. свой снимок окна -> screen_to_board (экран)
  4. сравнить ВЕРХНИЕ карты колонок, waste и фундаменты.

Любое расхождение = тот самый десинхрон, который раньше давал «Stuck».
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from act import controller as act
from orchestrator.rule_engine import Card, Suit
from vision import screen_to_board as s2b
from vision.recognizer import get_recognizer

BASE = "http://127.0.0.1:8000"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def post(path):
    req = urllib.request.Request(BASE + path, method="POST")
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def mem_tops(state: str):
    """Верхние карты колонок из render(_board) сервера."""
    cols = {}
    waste = None
    found = {}
    for line in state.splitlines():
        line = line.strip()
        if line.startswith("["):
            idx = int(line[1:line.index("]")]) - 1
            rest = line[line.index("]") + 1:].strip()
            if rest in ("(empty)", ""):
                cols[idx] = None
            else:
                cols[idx] = rest.split()[-1]
        elif line.startswith("Stock("):
            w = line.split("Waste:")[1].split("Foundations:")[0].strip()
            waste = None if w in ("-", "") else w
            f = line.split("Foundations:")[1].strip()
            for part in f.split():
                k, v = part.split(":")
                found[k] = None if v == "-" else v
    return cols, waste, found


def fmt(c: Card | None):
    from orchestrator.rule_engine import RANK_NAMES
    if c is None:
        return None
    return f"{RANK_NAMES.get(c.rank, c.rank)}{c.suit.value}"


rec = get_recognizer()
hwnd = act.find_game_window()
mismatch_total = 0

for step in range(N):
    r = post("/move")
    move = r.get("move")
    dump = get("/dump")
    state = dump["board"]

    img = act.capture_window(hwnd)
    H, W = img.shape[:2]
    fresh = s2b.screen_to_board(img, recognizer=rec)

    mcols, mwaste, mfound = mem_tops(state)
    problems = []
    for ci in range(7):
        mem_top = mcols.get(ci)
        ref_top = fmt(fresh.tableau[ci][-1]) if fresh.tableau[ci] else None
        if mem_top != ref_top:
            problems.append(f"col{ci+1}: mem={mem_top} screen={ref_top}")
    rw = fmt(fresh.waste[-1]) if fresh.waste else None
    if mwaste != rw:
        problems.append(f"waste: mem={mwaste} screen={rw}")
    for s in ("♥", "♦", "♣", "♠"):
        pile = fresh.foundation.get(next(x for x in Suit if x.value == s), [])
        rv = fmt(pile[-1]) if pile else None
        if mfound.get(s) != rv:
            problems.append(f"found {s}: mem={mfound.get(s)} screen={rv}")

    status = "OK " if not problems else "MISMATCH"
    mismatch_total += len(problems)
    print(f"[{step+1:2d}] {status} move={move}")
    if problems:
        for p in problems:
            print(f"       !! {p}")
    print(f"     reasoning: {r.get('reasoning','')[:150]}")
    print(f"     memory: {state.splitlines()[-7] if len(state.splitlines())>=7 else ''}"
          f" | {state.splitlines()[-1] if state else ''}")
    if r.get("won"):
        print("     🏆 ПОБЕДА")
        break
    time.sleep(1.0)

print(f"\nИТОГО расхождений память/экран: {mismatch_total}")
