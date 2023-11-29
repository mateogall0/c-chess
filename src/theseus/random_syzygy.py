#!/usr/bin/env python3
import chess
import chess.svg
import numpy as np


def count_pieces(board):
    total_pieces = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            total_pieces += 1

    return total_pieces


def random_fen(limit, board=chess.Board()):
    while 1:
        legal_moves = [move for move in board.legal_moves]
        if not legal_moves:
            break

        random_move = np.random.choice(legal_moves)
        board.push(random_move)
        pieces = count_pieces(board)
        if pieces <= limit:
            break
    return board.fen(), board.is_game_over(), pieces


if __name__ == '__main__':
    limit = 3
    for _ in range(5):
        fen, is_over, pieces = random_fen(limit=limit)
        limit += 1
        if limit > 7: limit = 3
        print(limit)
        print("Random FEN:", fen, is_over, pieces)
        print(chess.Board(fen))
