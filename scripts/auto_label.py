"""
Авторазметка уголков карт локальной vision-моделью (qwen2.5vl через Ollama).

Берёт каждую картинку из dataset/_unlabeled/, спрашивает модель "что за карта",
и переносит файл в dataset/<rank>_<suit>/ (например dataset/8_diamonds/).

НАМЕРЕННО не идеален: модель иногда путает масть (черви/буби, трефы/пики).
Поэтому после прогона нужно пройтись по папкам глазами и удалить неверные.
"""

import base64
import json
import sys
import urllib.request
from pathlib import Path

# ── 1. Настройки ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNLABELED_DIR = PROJECT_ROOT / "dataset" / "_unlabeled"
LABELED_ROOT = PROJECT_ROOT / "dataset"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:7b"

# ── 2. Словари-переводчики: что говорит модель -> что за папка ──────────
# Модель отвечает по-английски ("eight diamonds"), а папки у нас англ. тоже.
# Но на случай синонимов/опечаток держим словарь нормализации.
SUIT_MAP = {
    "hearts": "hearts",
    "heart": "hearts",
    "diamonds": "diamonds",
    "diamond": "diamonds",
    "clubs": "clubs",
    "club": "clubs",
    "spades": "spades",
    "spade": "spades",
}
RANK_MAP = {
    "ace": "A", "a": "A",
    "two": "2", "2": "2",
    "three": "3", "3": "3",
    "four": "4", "4": "4",
    "five": "5", "5": "5",
    "six": "6", "6": "6",
    "seven": "7", "7": "7",
    "eight": "8", "8": "8",
    "nine": "9", "9": "9",
    "ten": "10", "10": "10",
    "jack": "J", "j": "J",
    "queen": "Q", "q": "Q",
    "king": "K", "k": "K",
}

PROMPT = (
    "You see the top-left corner of a playing card showing its rank and suit. "
    "Answer with exactly two words: the rank word and the suit word. "
    "For example: 'ten clubs', 'queen hearts', 'ace spades'."
)


# ── 3. Функция: спросить модель про одну картинку ───────────────────────
def ask_model(image_path: Path) -> str:
    """Отправляет картинку модели и возвращает её ответ текстом."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False,
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["response"].strip()


# ── 4. Функция: превратить ответ модели в (rank, suit) ─────────────────
def parse_answer(answer: str):
    """
    Из строки типа "eight diamonds" делает ("8", "diamonds").
    Возвращает None, если распознать не удалось.
    """
    words = answer.lower().strip().split()
    if len(words) < 2:
        return None
    rank_word = words[0]
    suit_word = words[-1]  # берём последнее слово как масть
    rank = RANK_MAP.get(rank_word)
    suit = SUIT_MAP.get(suit_word)
    if rank is None or suit is None:
        return None
    return rank, suit


# ── 5. Главный цикл разметки ───────────────────────────────────────────
def label_all(limit: int | None = None):
    files = sorted(UNLABELED_DIR.glob("*.png"))
    if limit is not None:
        files = files[:limit]

    print(f"Всего картинок для разметки: {len(files)}")
    stats = {"ok": 0, "skipped": 0}

    for i, fpath in enumerate(files, start=1):
        try:
            answer = ask_model(fpath)
            parsed = parse_answer(answer)
        except Exception as e:
            print(f"[{i}/{len(files)}] {fpath.name}: ОШИБКА {e}")
            stats["skipped"] += 1
            continue

        if parsed is None:
            print(f"[{i}/{len(files)}] {fpath.name}: непонятный ответ {answer!r} — пропуск")
            stats["skipped"] += 1
            continue

        rank, suit = parsed
        target_dir = LABELED_ROOT / f"{rank}_{suit}"
        target_dir.mkdir(parents=True, exist_ok=True)
        fpath.rename(target_dir / fpath.name)
        stats["ok"] += 1
        print(f"[{i}/{len(files)}] {fpath.name} -> {rank}_{suit}   (ответ: {answer})")

    print(f"\nГотово. Размечено: {stats['ok']}, пропущено: {stats['skipped']}")


if __name__ == "__main__":
    # Если передали число аргументом — обработаем только столько картинок
    # (для теста). Иначе — все.
    limit = None
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    label_all(limit)
