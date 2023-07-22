#!/usr/bin/env python3


from tensorflow import keras as K
import numpy as np
import chess


def fen_to_bitboard(fen):
    board = chess.Board(fen)
    piece_to_int = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6, '.': 0}

    board_array = np.zeros((8, 8), dtype=int)
    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)
            piece = board.piece_at(square)
            if piece is not None:
                board_array[row, col] = piece_to_int[piece.symbol()]

    possible_moves = [move.uci() for move in board.legal_moves]
    moves_array = np.zeros((len(possible_moves), 64, 64), dtype=int)
    for idx, move in enumerate(possible_moves):
        from_square, to_square = chess.square(ord(move[0]) - ord('a'), 7 - (ord(move[1]) - ord('1'))), \
                                 chess.square(ord(move[2]) - ord('a'), 7 - (ord(move[3]) - ord('1')))
        moves_array[idx, from_square, to_square] = 1

    return board_array, moves_array



def new_model(lr=0.001):
    model = K.models.Sequential()
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dense(64, activation='linear'))
    model.compile(loss='mse', optimizer=K.optimizers.Adam(lr=lr))
    return model


if __name__ == '__main__':
    model = new_model()
    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    brd, moves_array = fen_to_bitboard(fen)
    move_idx = np.where(moves_array[:, 6*8 + 5, 5*8 + 5] == 1)[0][0]
    board = chess.Board(fen)
    print(move_idx)
