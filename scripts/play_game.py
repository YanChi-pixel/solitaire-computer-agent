"""
Прогон одной партии целиком, с записью reasoning LLM-стратега на каждом ходу
в текстовый лог. Это шаг 2 из README: "текстовый лог reasoning", ещё без
интерфейса — просто файл, который уже можно читать и вставлять в статью.

Использование:
    python3 play_game.py --seed 7 --max-moves 200
    python3 play_game.py --backend ollama --model qwen2.5:14b --seed 7
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Поднимаем корень проекта в sys.path, чтобы абсолютные импорты
# `orchestrator.*` / `planner.*` работали из любого cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from orchestrator.rule_engine import Board, apply_move, get_valid_moves, render
from planner.llm_planner import (
    LLMBackend,
    MockBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
    choose_move,
)
from planner.prompts import InvalidPlannerResponse

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def build_backend(args: argparse.Namespace) -> LLMBackend:
    if args.backend == "mock":
        return MockBackend()
    if args.backend == "ollama":
        return OllamaBackend(model_name=args.model, host=args.ollama_host)
    if args.backend == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit(
                "Не найден ключ. Экспортируйте его перед запуском:\n"
                "  export DEEPSEEK_API_KEY=sk-..."
            )
        return OpenAICompatibleBackend(
            model_name=args.model, api_key=api_key, base_url=DEEPSEEK_BASE_URL
        )
    raise ValueError(f"Неизвестный backend: {args.backend}")


def play_game(backend: LLMBackend, seed: int, max_moves: int, log_path: Path) -> dict:
    board = Board.new_game(seed=seed)
    log_lines: list[str] = [
        f"# Партия #{seed} — модель: {backend.model_name}",
        f"# Начало: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    outcome = "max_moves_reached"
    move_number = 0

    for move_number in range(1, max_moves + 1):
        if board.is_won():
            outcome = "won"
            break

        valid_moves = get_valid_moves(board)
        if not valid_moves:
            outcome = "stuck"
            break

        log_lines.append(f"## Ход {move_number}")
        log_lines.append("```")
        log_lines.append(render(board))
        log_lines.append("```")

        try:
            result = choose_move(backend, board, valid_moves)
        except InvalidPlannerResponse as e:
            # Модель ответила некорректно — фиксируем в логе и останавливаем партию,
            # а не подставляем случайный ход втихую: для бенчмарка важно видеть,
            # что модель реально сломалась, а не просто "доиграла".
            log_lines.append(f"⚠️ Модель дала невалидный ответ: {e}")
            outcome = "planner_error"
            break

        chosen_move = valid_moves[result.move_index]
        log_lines.append(f"🧠 Reasoning: {result.reasoning}")
        log_lines.append(f"➡️ Ход: {chosen_move}")
        log_lines.append("")

        board = apply_move(board, chosen_move)

    log_lines.append(f"# Итог: {outcome}, ходов сделано: {move_number}")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    return {
        "outcome": outcome,
        "moves_made": move_number,
        "model_name": backend.model_name,
        "seed": seed,
        "log_path": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Прогон партии в Косынку с LLM-стратегом")
    parser.add_argument("--seed", type=int, default=7, help="Seed для раскладки")
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--backend", choices=["mock", "ollama", "deepseek"], default="mock")
    parser.add_argument(
        "--model", default=None,
        help="Название модели. По умолчанию: qwen2.5:14b для ollama, "
             "deepseek-chat для deepseek",
    )
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    if args.model is None:
        args.model = "deepseek-chat" if args.backend == "deepseek" else "qwen2.5:14b"

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"game_seed{args.seed}_{args.backend}.md"

    backend = build_backend(args)
    summary = play_game(backend, args.seed, args.max_moves, log_path)

    print(f"Итог: {summary['outcome']}")
    print(f"Ходов сделано: {summary['moves_made']}")
    print(f"Лог сохранён: {summary['log_path']}")


if __name__ == "__main__":
    main()
