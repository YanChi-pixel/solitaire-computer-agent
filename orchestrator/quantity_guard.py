"""
Quantity guard: закрывает дыру, которую оставляет verify_board_tops — он
проверяет ТОЛЬКО идентичность верхней карты, никогда не количество карт в
стопке.

ВАЖНО (v2, после реального провала на живой партии): калибровка и проверка
идут ТОЛЬКО по DESTINATION-колонке хода, никогда по source. Причина —
структурная, не статистическая: source-колонка после снятия карт может либо
просто укоротиться (сдвиг = N*fan_step, чистый сигнал), либо ВСКРЫТЬ рубашку
под собой (сдвиг — смесь fan_step открытых и шага рубашек, грязный сигнал,
две физически разные величины в одном числе). Destination-колонка ТАКОГО
никогда не делает — вскрытие рубашки возможно только там, откуда карту
убрали, а не туда, куда её положили. Поэтому один и тот же тип наблюдения
(N*fan_step) гарантированно чист на destination, и его не нужно фильтровать
постфактум — грязных наблюдений там просто не бывает по конструкции игры.

Как источник калибровки используется ЛЮБОЙ уже подтверждённый (verify_board_tops
== ok) ход tableau_to_tableau/waste_to_tableau, а не только card_count==1 —
раз сигнал чист, дополнительно ограничивать себя одиночными ходами незачем,
это только медленнее собирает данные.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass
class FanStepEstimator:
    """Живая оценка шага веера ОТКРЫТЫХ карт (px), по destination-наблюдениям."""
    samples: list[int] = field(default_factory=list)
    max_samples: int = 20

    def observe(self, delta_px: int, card_count: int) -> None:
        """
        Вызывать после КАЖДОГО подтверждённого хода, где card_count карт
        приземлились на destination-колонку (неважно, 1 их было или больше —
        сигнал одинаково чист). Нормализуем на card_count, чтобы копить
        оценки именно ШАГА ОДНОЙ карты, а не разномасштабные сдвиги.
        """
        if card_count <= 0:
            return
        per_card = abs(delta_px) / card_count
        self.samples.append(per_card)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

    @property
    def value(self) -> int | None:
        if not self.samples:
            return None
        return round(median(self.samples))


def verify_quantity(
    dest_open_y_before: int | None,
    dest_open_y_after: int | None,
    expected_card_count: int,
    fan_step: int | None,
    table_top_y: int,
    tolerance: int = 6,
) -> tuple[bool, str]:
    """
    Проверяет DESTINATION-колонку хода (НЕ source — см. docstring модуля).

    dest_open_y_before/after — open_y колонки-цели до/после хода.
    table_top_y нужен для одного особого случая: цель была ПУСТОЙ колонкой
    (dest_open_y_before is None, ход — король на пустую колонку). Тогда
    сравнивать нечего дельтой — сравниваем АБСОЛЮТНУЮ позицию: после хода
    верхняя открытая карта веера должна начинаться ровно на
    table_top_y + (card_count - 1) * fan_step (первая карта веера стоит у
    самого верха колонки, остальные — каскадом ниже неё).
    """
    if fan_step is None:
        return True, "fan_step not yet calibrated this session — skipped"

    if dest_open_y_after is None:
        return False, "destination column has no open card after the move — did it land at all?"

    if dest_open_y_before is None:
        # цель была пустой колонкой — сравниваем абсолютную позицию, не дельту
        expected_abs = table_top_y + (expected_card_count - 1) * fan_step
        if abs(dest_open_y_after - expected_abs) > tolerance:
            return False, (
                f"empty-column landing: open_y={dest_open_y_after}, "
                f"expected {expected_abs} for {expected_card_count} card(s)"
            )
        return True, "ok (empty-column landing)"

    expected_shift = expected_card_count * fan_step
    actual_shift = abs(dest_open_y_after - dest_open_y_before)
    if abs(actual_shift - expected_shift) > tolerance:
        return False, (
            f"destination open_y shifted {actual_shift}px, expected {expected_shift}px "
            f"for {expected_card_count} card(s) at fan_step={fan_step}px"
        )
    return True, "ok"
