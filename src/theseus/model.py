#!/usr/bin/env python3


from tensorflow import keras as K
import tensorflow as tf
import numpy as np
import chess
max_moves = 64
files = {
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8,
}
files_reversed = 'abcdefgh'


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
    return board_array


def new_model(lr=0.001):
    board_input_layer = K.layers.Input(shape=(8, 8))
    moves_input_layer = K.layers.Input(shape=(1, max_moves))

    board_flattened = K.layers.Flatten()(board_input_layer)
    moves_flattened = K.layers.Flatten()(moves_input_layer)

    merged_inputs = K.layers.concatenate([board_flattened, moves_flattened])

    hidden_layer = K.layers.Dense(356, activation='relu')(merged_inputs)
    output_layer = K.layers.Dense(max_moves, activation='softmax')(hidden_layer)

    model = K.models.Model(inputs=[board_input_layer, moves_input_layer], outputs=output_layer)
    model.compile(loss='mse', optimizer=K.optimizers.Adam(lr=lr))
    return model



def make_move(model, fen):
    board = chess.Board(fen)
    possible_moves = [0 for _ in range(max_moves)]
    for c, i in enumerate(board.legal_moves):
        current_move = 0
        for j, item in enumerate(str(i)):
            if j % 2 == 0:
                current_move += 10 ** j * files[item]
                continue
            current_move += 10 ** j * int(item)
        possible_moves[c] = current_move
    print(np.shape(possible_moves))
    print(possible_moves)
    brd= fen_to_bitboard(fen)
    output = model.predict(([brd], [[possible_moves]]))
    print(output)
    chosen_move_index = np.argmax(output)
    print(chosen_move_index)
    chosen = possible_moves[chosen_move_index]
    if chosen == 0:
        return 0
    a = files_reversed[chosen // 1000]
    b = str(chosen // 100 % 10)
    c = files_reversed[chosen // 10 % 10]
    d = str(chosen % 10)
    return a + b + c + d


if __name__ == '__main__':
    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

    model = new_model()

    print(make_move(model, fen))
