#!/usr/bin/env python3


import chess
from theseus.model import make_move as mMakeMove
from tensorflow import keras as K
model = K.models.load_model('theseus/theseus.h5')

def print_board(fen):
    """
    FEN parts: board, active color, castling rights,
    en passant target square, halfmove clock, and fullmove number
    """
    parts = fen.split()
    board_rows = parts[0].split('/')

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
    print("    a   b   c   d   e   f   g   h")


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


def check_game_state(fen):
    board = chess.Board(fen)
    finished = False
    if board.is_checkmate():
        print("Checkmate!")
        finished = True
    elif board.is_stalemate():
        print("Stalemate!")
        finished = True
    elif board.is_insufficient_material():
        print("Insufficient material")
        finished = True
    elif board.is_seventyfive_moves():
        print("Fifty-moves")
        finished = True
    elif board.is_repetition():
        print("Three-fold repetition")
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


def play(fen, turn=0, engine_only=False):
    while not check_game_state(fen):
        print_board(fen)
        if not engine_only and turn % 2 == 0:
            move = input(f"Move ({get_turn(fen)}): ")
        else:
            _, move, _ = mMakeMove(model, fen)
        if is_move_possible(fen, str(move)):
            fen = make_move(fen, str(move))
        else:
            print('Move not possible')
        turn += 1
    print_board(fen)


if __name__ == '__main__':
    # Initial position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
    fen = 'rnbqkbnr/pppp1ppp/8/8/4Pp2/5N2/PPPP2PP/RNBQKB1R b KQkq - 1 3'
    play(fen, engine_only=True)