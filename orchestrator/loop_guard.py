"""
Loop guard: отсеивает ходы tableau_to_tableau, которые возвращают расклад
колонок в уже посещённое за эту партию состояние — то есть провально
детерминированно ловит ЛЮБОЙ цикл (не только пару/тройку подряд), потому
что опирается на факт повтора состояния, а не на форму последовательности
ходов.

Место в архитектуре: это СЕРВЕРНАЯ, стейтфул-логика (нужна история за
партию), поэтому она не в rule_engine.py (который остаётся чистой функцией
без памяти о прошлом) — ровно то же разделение ответственности, что и
везде в этом проекте.
"""

from __future__ import annotations

from orchestrator.rule_engine import Board, Move, apply_move


def tableau_signature(board: Board) -> tuple:
    """
    Хешируемый снимок ТОЛЬКО расклада колонок (без waste/stock/foundation).
    Face-down карта — фиксированный маркер (UNKNOWN_RANK, независимо от
    того, узнаем ли мы её потом), поэтому первое вскрытие карты ВСЕГДА меняет
    сигнатуру (маркер -> настоящая карта) и не может быть ложно принято за
    повтор, а любое чистое перекладывание уже открытых карт без вскрытия
    нового — наоборот, корректно даёт совпадающую сигнатуру.
    """
    return tuple(
        tuple((c.rank, c.suit, c.face_up) for c in col)
        for col in board.tableau
    )


class LoopGuard:
    def __init__(self):
        self._visited: set[tuple] = set()
        self.stalled_recycles = 0  # полных проходов колоды подряд без прогресса

    def reset_progress(self):
        """Вызывать при ЛЮБОМ ходе на foundation — это необратимый прогресс,
        после которого старые раскланды колонок разрешено проходить заново."""
        self._visited.clear()
        self.stalled_recycles = 0

    def note_recycle_without_progress(self):
        self.stalled_recycles += 1

    def is_stuck(self, max_recycles_without_progress: int = 2) -> bool:
        """Партия объективно застряла: колода прошла N раз подряд без
        единого хода на foundation. Это не баг детектора — часть реальных
        раскладов Косынки действительно нерешаема, и явно объявить это
        лучше, чем крутить демо вечно."""
        return self.stalled_recycles >= max_recycles_without_progress

    def filter_progressing_moves(self, board: Board, moves: list[Move]) -> list[Move]:
        """
        Убирает из списка ходов те tableau_to_tableau, которые ведут в уже
        посещённый расклад колонок. draw НИКОГДА не фильтруется — тасование
        колоды не меняет расклад колонок по определению, и это ожидаемо
        (иначе draw всегда бы ложно считался "повтором").
        """
        current_sig = tableau_signature(board)
        self._visited.add(current_sig)

        kept = []
        for move in moves:
            if move.kind != "tableau_to_tableau":
                kept.append(move)
                continue
            resulting_board = apply_move(board, move)
            sig = tableau_signature(resulting_board)
            if sig not in self._visited:
                kept.append(move)
        return kept
