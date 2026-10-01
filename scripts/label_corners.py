"""
Разметка собранных уголков — ЗАПУСКАЕТСЯ НА ВАШЕМ КОМПЬЮТЕРЕ (нужен показ
картинки). Берёт всё из dataset/_unlabeled/, показывает по одной, спрашивает
ранг+масть, перекладывает в dataset/<rank>_<suit>/.

Это НАМЕРЕННО не автоматическая разметка — человек в контуре обязателен:
автоматическое извлечение уголков иногда ошибается в кадрировании (мы это
уже видели на реальных скриншотах — где-то виден только символ масти без
ранга), и такие плохие примеры лучше отсеять здесь, а не кормить ими сеть.

Ввод: одна буква/цифра ранга + одна буква масти, например "10 c", "q h".
Пропустить нечитаемый уголок: пустой ввод.
Выйти: "q" как весь ввод (не путать с рангом "Q"! используется "quit" целиком).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

# Абсолютные пути от корня проекта, чтобы разметка работала из любого cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

UNLABELED_DIR = _PROJECT_ROOT / "dataset" / "_unlabeled"
LABELED_ROOT = _PROJECT_ROOT / "dataset"

SUIT_NAMES = {"h": "hearts", "d": "diamonds", "c": "clubs", "s": "spades"}
RANK_NAMES = {
    "a": "A", "j": "J", "q": "Q", "k": "K",
    **{str(n): str(n) for n in range(2, 11)},
}


def parse_label(raw: str) -> tuple[str, str] | None:
    """'10 c' -> ('10', 'clubs'); 'q h' -> ('Q', 'hearts'); некорректный ввод -> None."""
    parts = raw.strip().lower().split()
    if len(parts) != 2:
        return None
    rank_raw, suit_raw = parts
    if rank_raw not in RANK_NAMES or suit_raw not in SUIT_NAMES:
        return None
    return RANK_NAMES[rank_raw], SUIT_NAMES[suit_raw]


def label_all(unlabeled_dir: Path = UNLABELED_DIR, show_images: bool = True) -> dict:
    stats = {"labeled": 0, "skipped": 0, "quit": False}
    files = sorted(unlabeled_dir.glob("*.png"))
    print(f"Найдено {len(files)} неразмеченных уголков.")

    for i, fpath in enumerate(files):
        if show_images:
            Image.open(fpath).show()

        raw = input(f"[{i+1}/{len(files)}] {fpath.name} — ранг масть (или Enter=пропустить, 'quit'=выход): ")
        if raw.strip().lower() == "quit":
            stats["quit"] = True
            break

        if not raw.strip():
            stats["skipped"] += 1
            continue

        parsed = parse_label(raw)
        if parsed is None:
            print("  Не поняла формат, пропускаю. Пример правильного ввода: '10 c' или 'q h'")
            stats["skipped"] += 1
            continue

        rank, suit = parsed
        target_dir = LABELED_ROOT / f"{rank}_{suit}"
        target_dir.mkdir(parents=True, exist_ok=True)
        fpath.rename(target_dir / fpath.name)
        stats["labeled"] += 1

    return stats


if __name__ == "__main__":
    result = label_all()
    print(f"\nРазмечено: {result['labeled']}, пропущено: {result['skipped']}")
