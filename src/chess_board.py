#!/usr/bin/env python3


import chess


def print_board(fen):
    """
    FEN parts: board, active color, castling rights,
    en passant target square, halfmove clock, and fullmove number
    """
    parts = fen.split()
    board_rows = parts[0].split('/')

    print("    a   b   c   d   e   f   g   h")
    print("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")

    for i, row in enumerate(board_rows):
        # Translate FEN row to board row
        board_row = ""
        for char in row:
            if char.isdigit():
                # Add empty squares
                board_row += ' ' * int(char)
            else:
                # Add piece
                board_row += char

        print(f"{8 - i} │ " + " │ ".join(list(board_row)) + " │")

        if i != len(board_rows) - 1:
            print("  ├───┼───┼───┼───┼───┼───┼───┼───┤")

    print("  └───┴───┴───┴───┴───┴───┴───┴───┘")


def is_move_possible(fen, move):
    board = chess.Board(fen)
    try:
        move_obj = chess.Move.from_uci(move)

        return move_obj in board.legal_moves
    except Exception:
        return False


def make_move(fen, move):
    board = chess.Board(fen)
    move_obj = chess.Move.from_uci(move)
    board.push(move_obj)
    return board.fen()


def check_game_state(board):

    finished = False
    if board.is_checkmate():
        print("Checkmate!")
        finished = True
    elif board.is_stalemate():
        print("Stalemate!")
        finished = True
    else:
        print("Game is still in progress.")
    return finished


def get_legal_moves(fen):
    board = chess.Board(fen)
    legal_moves_generator = board.generate_legal_moves()
    return [move.uci() for move in legal_moves_generator]


def get_turn(fen):
    board = chess.Board(fen)
    return "white" if board.turn == chess.WHITE else "black"


def play(fen):
    while not check_game_state(fen):
        print_board(fen)
        move = input(f"Move ({get_turn(fen)}): ")
        if is_move_possible(fen, move):
            fen = make_move(fen, move)
        else:
            print('Move not possible')
    print_board(fen)
    


if __name__ == '__main__':
    # Initial position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    play(fen)