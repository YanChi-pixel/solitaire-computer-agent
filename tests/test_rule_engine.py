import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# UTF-8 stdout, чтобы символы мастей не падали в консоли Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from orchestrator.rule_engine import (
    Board,
    Card,
    Move,
    Suit,
    apply_move,
    full_deck,
    get_valid_moves,
    render,
)


def test_new_game_has_28_tableau_cards_and_24_in_stock():
    board = Board.new_game(seed=42)
    total_tableau = sum(len(col) for col in board.tableau)
    assert total_tableau == 28  # 1+2+3+4+5+6+7
    assert len(board.stock) == 24


def test_only_last_card_in_each_column_is_face_up():
    board = Board.new_game(seed=1)
    for col in board.tableau:
        for card in col[:-1]:
            assert not card.face_up
        assert col[-1].face_up


def test_draw_move_always_available_at_game_start():
    board = Board.new_game(seed=1)
    moves = get_valid_moves(board)
    assert any(m.kind == "draw" for m in moves)


def test_king_can_move_to_empty_column_only():
    board = Board()
    board.tableau[0] = [Card(13, Suit.SPADES, face_up=True)]  # король открыт
    board.tableau[1] = []  # пустая колонка
    board.tableau[2] = [Card(5, Suit.HEARTS, face_up=True)]

    moves = get_valid_moves(board)
    king_moves = [
        m for m in moves
        if m.kind == "tableau_to_tableau" and m.from_index == 0
    ]
    targets = {m.to_index for m in king_moves}
    assert 1 in targets      # можно на пустую
    assert 2 not in targets  # нельзя на пятёрку


def test_alternating_color_stacking_rule():
    board = Board()
    board.tableau[0] = [Card(9, Suit.CLUBS, face_up=True)]   # 9 чёрная
    board.tableau[1] = [Card(8, Suit.HEARTS, face_up=True)]  # 8 красная -> можно на 9 чёрную
    board.tableau[2] = [Card(8, Suit.SPADES, face_up=True)]  # 8 чёрная -> нельзя на 9 чёрную

    moves = get_valid_moves(board)
    valid_targets = {
        m.to_index for m in moves
        if m.kind == "tableau_to_tableau" and m.from_index == 1
    }
    assert 0 in valid_targets

    invalid_targets = {
        m.to_index for m in moves
        if m.kind == "tableau_to_tableau" and m.from_index == 2
    }
    assert 0 not in invalid_targets


def test_ace_to_empty_foundation_and_sequential_build():
    board = Board()
    board.tableau[0] = [Card(1, Suit.HEARTS, face_up=True)]
    moves = get_valid_moves(board)
    assert any(m.kind == "tableau_to_foundation" for m in moves)

    ace_move = next(m for m in moves if m.kind == "tableau_to_foundation")
    board = apply_move(board, ace_move)
    assert board.foundation[Suit.HEARTS][-1].rank == 1

    board.tableau[1] = [Card(2, Suit.HEARTS, face_up=True)]
    moves = get_valid_moves(board)
    assert any(
        m.kind == "tableau_to_foundation" and m.from_index == 1 for m in moves
    )


def test_apply_move_does_not_mutate_original_board():
    board = Board.new_game(seed=5)
    moves = get_valid_moves(board)
    draw_move = next(m for m in moves if m.kind == "draw")
    original_stock_len = len(board.stock)
    new_board = apply_move(board, draw_move)
    assert len(board.stock) == original_stock_len  # исходный объект не тронут
    # draw-3: один клик берёт до 3 карт из stock (24 -> 21).
    assert len(new_board.stock) == original_stock_len - 3
    assert len(new_board.waste) == 3


def test_flipping_new_top_card_after_moving_from_tableau():
    board = Board()
    board.tableau[0] = [
        Card(10, Suit.CLUBS, face_up=False),
        Card(9, Suit.HEARTS, face_up=True),
    ]
    board.tableau[1] = [Card(10, Suit.SPADES, face_up=True)]

    moves = get_valid_moves(board)
    move = next(
        m for m in moves
        if m.kind == "tableau_to_tableau" and m.from_index == 0 and m.to_index == 1
    )
    new_board = apply_move(board, move)
    assert new_board.tableau[0][-1].face_up  # десятка треф открылась


def test_full_deck_has_52_unique_cards():
    deck = full_deck()
    assert len(deck) == 52
    assert len(set((c.rank, c.suit) for c in deck)) == 52


def test_draw_append_preserves_total_stock_plus_waste():
    # Критерий из разбора: len(stock)+len(waste) обязано оставаться
    # константой между draw-ходами. Если сумма проседает — это НАСТОЯЩАЯ
    # утечка карт (баг замены "окно из 3"), а не законный рост кучи waste.
    # append-модель должна держать сумму ровно 24 весь проход колоды.
    board = Board.new_game(seed=42)
    moves = get_valid_moves(board)
    draw = next(m for m in moves if m.kind == "draw")

    total = len(board.stock) + len(board.waste)
    assert total == 24

    # Двигаем через весь проход: 24/3 = 8 draw до опустошения stock.
    for _ in range(8):
        board = apply_move(board, draw)
        assert len(board.stock) + len(board.waste) == 24, (
            "карты потерялись: сумма stock+waste изменилась"
        )

    # К концу прохода stock пуст, waste копит всю историю (24 карты).
    assert len(board.stock) == 0
    assert len(board.waste) == 24

    # Recycle: весь waste возвращается в stock, порядок сохраняется.
    board = apply_move(board, draw)
    assert len(board.stock) == 24
    assert len(board.waste) == 0


def test_draw_append_does_not_drop_unplayed_remainder():
    # Прямая проверка бага "замены": при новом draw непройденный остаток
    # прошлой тройки НЕ должен исчезать. Ставим в waste верхнюю известную
    # карту и убеждаемся, что после draw она остаётся в куче (ниже новых).
    board = Board.new_game(seed=7)
    # форсируем: верхняя карта waste уже "известна" (как после _sync_revealed_card)
    known = Card(13, Suit.HEARTS, face_up=True)
    board.waste = [known]
    before_total = len(board.stock) + len(board.waste)

    draw = Move(kind="draw")
    board = apply_move(board, draw)

    assert known in board.waste, "непройденный остаток waste исчез при draw"
    assert len(board.stock) + len(board.waste) == before_total


def _run_random_playout(seed: int, max_moves: int = 500) -> tuple[int, bool]:
    """Играет случайными валидными ходами, пока не застрянет или не выиграет."""
    board = Board.new_game(seed=seed)
    for i in range(max_moves):
        if board.is_won():
            return i, True
        moves = get_valid_moves(board)
        if not moves:
            return i, False
        import random
        move = random.Random(seed + i).choice(moves)
        board = apply_move(board, move)
    return max_moves, False


def test_random_playout_runs_without_crashing():
    # Не про то, чтобы выиграть — просто движок не должен падать
    # на сотнях случайных ходов ни в одной из партий.
    for seed in range(20):
        moves_made, won = _run_random_playout(seed, max_moves=300)
        assert moves_made > 0


if __name__ == "__main__":
    # Быстрая ручная демонстрация: печатаем стол и первые доступные ходы
    board = Board.new_game(seed=7)
    print(render(board))
    print()
    print(f"Доступно ходов: {len(get_valid_moves(board))}")
    for m in get_valid_moves(board)[:8]:
        print(" ", m)
