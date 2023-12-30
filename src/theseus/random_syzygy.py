#!/usr/bin/env python3
"""
This module contains a random Chess position generator with a count of
pieces that goes from 3 to 7. The objective is to obtain unbiased new data
as validation for the Theseus model.
"""
import chess
import chess.svg
import numpy as np
import requests
from theseus import Theseus

files = {
    'a': '1',
    'b': '2',
    'c': '3',
    'd': '4',
    'e': '5',
    'f': '6',
    'g': '7',
    'h': '8',
}

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

def random_syzygy(verbose=True, iterations=14):
    fen_codes_readable = []
    for _ in range(iterations):
        fen, is_over, _ = random_fen()
        if not is_over:
            fen_codes_readable.append(fen)
    fen_codes_readable = remove_redundancies(fen_codes_readable)
    fen_codes = [f.replace(' ', '_') for f in fen_codes_readable]
    if verbose:
        print(fen_codes)
        print(len(fen_codes))
    return fen_codes, fen_codes_readable

def get_syzygy_output(fen_codes=[], fen_codes_readable=[],
                      url='http://tablebase.lichess.ovh/standard?fen=',
                      verbose=True):
    Y = []
    X0 = []
    X1 = []
    X2 = []
    for i in range(len(fen_codes)):
        res = requests.get(url + fen_codes[i]).json()
        cat = res['category']
        if cat != 'win' and cat != 'cursed-win' and cat != 'maybe-win' and cat != 'draw':
            continue
        board = chess.Board(fen_codes_readable[i])
        who_moves = 1 if board.turn == chess.WHITE else 0
        X0.append(who_moves)
        X1.append(Theseus.board_to_bitboard(fen=fen_codes_readable[i]))
        possible_moves = [0 for _ in range(Theseus.max_moves)]
        for c, moves in enumerate(board.legal_moves):
            current_move = 0
            for j, item in enumerate(str(moves)):
                if j % 2 == 0:
                    try:
                        current_move += 10 ** j * Theseus.files[item]
                    except KeyError:
                        pass
                    continue
                current_move += 10 ** j * int(item)
            possible_moves[c] = current_move
        X2.append(possible_moves)
        uci_s = res['moves'][0]['uci'][:4]
        chosen_move = np.zeros((Theseus.max_moves,))
        for j, item in enumerate(board.legal_moves):
            if uci_s == str(item)[:4]:
                chosen_move[j] = 1
                break
        else:
            print(f'Fen: {fen_codes[i]}')
            print(f'Fen readable: {fen_codes_readable[i]}')
            raise ValueError(f'{uci_s} not found in legal moves: {list(board.legal_moves)}')
        Y.append(chosen_move)

    if verbose:
        print(f'X0 len: {len(X0)}')
        print(f'X1 len: {len(X1)}')
        print(f'X2 len: {len(X2)}')
        print(f'Y  len: {len(Y)}')
    return (
        np.array(X0),
        np.array(X1),
        np.array(X2),
        np.array(Y),
    )


if __name__ == '__main__':
    fen_codes, fen_codes_readable = random_syzygy()
    X0, X1, X2, Y = get_syzygy_output(fen_codes, fen_codes_readable)
    np.savez('val_data/syzygy.npz', X0=X0, X1=X1, X2=X2, Y=Y)
