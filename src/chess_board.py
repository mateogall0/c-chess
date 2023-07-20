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
    return board.fen()


def check_game_state(fen):
    board = chess.Board(fen)

    if board.is_checkmate():
        print("Checkmate!")
    elif board.is_stalemate():
        print("Stalemate!")
    else:
        print("Game is still in progress.")


def get_possible_moves(fen):
    board = chess.Board(fen)
    legal_moves_generator = board.generate_legal_moves()
    return [move.uci() for move in legal_moves_generator]
    



if __name__ == '__main__':
    fen = 'rnbqkbnr/1ppp1ppp/p7/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 3'
    check_game_state(fen)
    print(get_possible_moves(fen))
