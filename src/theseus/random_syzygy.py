#!/usr/bin/env python3
import chess
import chess.svg
import random


def count_pieces(board):
    total_pieces = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            total_pieces += 1

    return total_pieces


def random_fen(limit=random.randint(3, 7), board=chess.Board()):
    while 1:
        legal_moves = [move for move in board.legal_moves]
        if not legal_moves:
            break

        random_move = random.choice(legal_moves)
        board.push(random_move)
        if count_pieces(board) <= limit:
            break
    return board.fen(), board.is_game_over()


random_fen = random_fen()
print("Random FEN:", random_fen)