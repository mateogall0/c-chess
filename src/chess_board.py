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
    move_obj = chess.Move.from_uci(move)

    return move_obj in board.legal_moves


def make_move(fen, move):
    board = chess.Board(fen)
    move_obj = chess.Move.from_uci(move)
    board.push(move_obj)
    updated_fen = board.fen()

    return updated_fen

def check_game_state(fen):
    board = chess.Board(fen)

    if board.is_checkmate():
        print("Checkmate!")
    elif board.is_stalemate():
        print("Stalemate!")
    else:
        print("Game is still in progress.")


if __name__ == '__main__':
    fen_checkmate = "rnbqkbnr/ppp1pQp1/7p/8/2Bp4/4P3/PPPP1PPP/RNB1K1NR b KQkq - 0 5"
    fen_stalemate = "8/8/8/8/8/8/1k6/K7 w - - 0 1"
    fen_in_progress = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    check_game_state(fen_checkmate)
    check_game_state(fen_stalemate)
    check_game_state(fen_in_progress)
