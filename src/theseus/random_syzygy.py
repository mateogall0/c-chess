#!/usr/bin/env python3
import chess
import chess.svg
import numpy as np
import  requests


def count_pieces(board):
    total_pieces = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            total_pieces += 1

    return total_pieces

def remove_redundancies(arr):
    return list(set(arr))

def random_fen():
    limit=np.random.randint(3, 8)
    board = chess.Board()
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

def random_syzygy(verbose=False, iterations=1400):
    fen_codes = []
    for _ in range(iterations):
        fen, is_over, _ = random_fen()
        if not is_over:
            fen_codes.append(fen.replace(' ', '_'))
    fen_codes = remove_redundancies(fen_codes)
    if verbose:
        print(fen_codes)
        print(len(fen_codes))
    return fen_codes

def get_syzygy_output(fen_codes=[], url='http://tablebase.lichess.ovh/standard?fen='):
    for fen in fen_codes:
        response = requests.get(url + fen)
        print(response)


if __name__ == '__main__':
    get_syzygy_output(random_syzygy(True))
