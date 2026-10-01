"""
Sort FULL card crops into <rank>_<suit> folders, using the local vision model.

For each image in dataset_full/_unlabeled/:
  1. Ask qwen2.5vl: "is this a single card, or a stack of 2-3 cards?"
  2. If STACK -> move to dataset_full/_trash/.
  3. If SINGLE -> ask for rank+suit, move to dataset_full/<rank>_<suit>/.

Nothing is deleted — stacks go to _trash so you can review them later.

Run:  python scripts/sort_full.py
"""

import base64
import json
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

UNLABELED = _ROOT / "dataset_full" / "_unlabeled"
TRASH = _ROOT / "dataset_full" / "_trash"
LABELED_ROOT = _ROOT / "dataset_full"

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "qwen2.5vl:7b"

RANK_MAP = {
    "ace": "A", "a": "A", "two": "2", "2": "2", "three": "3", "3": "3",
    "four": "4", "4": "4", "five": "5", "5": "5", "six": "6", "6": "6",
    "seven": "7", "7": "7", "eight": "8", "8": "8", "nine": "9", "9": "9",
    "ten": "10", "10": "10", "jack": "J", "j": "J", "queen": "Q", "q": "Q",
    "king": "K", "k": "K",
}
SUIT_MAP = {
    "hearts": "hearts", "heart": "hearts", "h": "hearts",
    "diamonds": "diamonds", "diamond": "diamonds", "d": "diamonds",
    "clubs": "clubs", "club": "clubs", "c": "clubs",
    "spades": "spades", "spade": "spades", "s": "spades",
}


def ask_vision(image_path: Path, prompt: str) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {"model": VISION_MODEL, "prompt": prompt, "images": [b64], "stream": False}
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip().lower()


def is_stack(image_path: Path) -> bool:
    prompt = (
        "Look at this image. Does it show exactly ONE playing card, or a STACK "
        "of two or more cards (where a second card's edge peeks out below the top "
        "card)? Answer with just one word: 'single' or 'stack'."
    )
    ans = ask_vision(image_path, prompt)
    return "stack" in ans or "multiple" in ans


def get_card(image_path: Path):
    prompt = (
        "Look at this playing card. What is its rank and suit? Answer with exactly "
        "two words: rank then suit. For example 'ten clubs' or 'queen hearts'."
    )
    ans = ask_vision(image_path, prompt)
    words = ans.split()
    rank = None
    suit = None
    for w in words:
        if rank is None and w in RANK_MAP:
            rank = RANK_MAP[w]
        elif suit is None and w in SUIT_MAP:
            suit = SUIT_MAP[w]
    return rank, suit


def main():
    TRASH.mkdir(parents=True, exist_ok=True)
    files = sorted(UNLABELED.glob("*.png"))
    print(f"Файлов для сортировки: {len(files)}")

    n_single = 0
    n_stack = 0
    n_fail = 0

    for i, f in enumerate(files, 1):
        try:
            if is_stack(f):
                f.rename(TRASH / f.name)
                n_stack += 1
                print(f"[{i}/{len(files)}] СТОПКА -> _trash: {f.name}")
                continue
        except Exception as e:
            print(f"[{i}/{len(files)}] ошибка проверки {f.name}: {e}")
            n_fail += 1
            continue

        try:
            rank, suit = get_card(f)
        except Exception as e:
            print(f"[{i}/{len(files)}] ошибка распознавания {f.name}: {e}")
            n_fail += 1
            continue

        if rank is None or suit is None:
            n_fail += 1
            print(f"[{i}/{len(files)}] не распознано, пропуск: {f.name}")
            continue

        target = LABELED_ROOT / f"{rank}_{suit}"
        target.mkdir(parents=True, exist_ok=True)
        f.rename(target / f.name)
        n_single += 1
        print(f"[{i}/{len(files)}] {rank}_{suit}: {f.name}")

    print(f"\nГотово. Одиночных: {n_single}, стопок: {n_stack}, не распознано: {n_fail}")


if __name__ == "__main__":
    main()
