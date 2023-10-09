#!/usr/bin/env python3


from tensorflow import keras as K
import tensorflow as tf
import numpy as np
import chess
max_moves = 86
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
tf.compat.v1.disable_eager_execution()

def board_to_bitboard(fen):
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


def new_model(lr=0.05):
    # Input of either true or false taking into consideration if the move is for white or black    
    color_input_layer = K.layers.Input(shape=(1,), dtype='int32')
    # Board input
    board_input_layer = K.layers.Input(shape=(8, 8))
    # Moves available in current position 
    moves_input_layer = K.layers.Input(shape=(1, max_moves))

    board_flattened = K.layers.Flatten()(board_input_layer)
    moves_flattened = K.layers.Flatten()(moves_input_layer)

    hidden_layer_board = K.layers.Dense(64, activation='relu')(board_flattened)
    hidden_layer_moves = K.layers.Dense(max_moves, activation='relu')(moves_flattened)
    hidden_layer_color = K.layers.Dense(32, activation='relu')(K.layers.Flatten()(K.layers.Embedding(2, 32)(color_input_layer)))

    merged_inputs = K.layers.concatenate([hidden_layer_color, hidden_layer_board, hidden_layer_moves])

    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(merged_inputs)
    hidden_layer = K.layers.BatchNormalization()(hidden_layer)
    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(hidden_layer)
    hidden_layer = K.layers.BatchNormalization()(hidden_layer)
    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(hidden_layer)
    hidden_layer = K.layers.BatchNormalization()(hidden_layer)
    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(hidden_layer)
    hidden_layer = K.layers.BatchNormalization()(hidden_layer)
    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(hidden_layer)
    hidden_layer = K.layers.BatchNormalization()(hidden_layer)
    hidden_layer = K.layers.Dense(64 + max_moves + 32, activation='relu')(hidden_layer)
    hidden_layer = K.layers.BatchNormalization()(hidden_layer)
    output_layer = K.layers.Dense(max_moves, activation='softmax')(hidden_layer)

    model = K.models.Model(inputs=[color_input_layer, board_input_layer, moves_input_layer], outputs=output_layer)
    model.compile(loss='mse', optimizer=K.optimizers.legacy.Adam(lr=lr), metrics=['accuracy'])
    return model


def make_move(model, fen):
    board = chess.Board(fen)
    possible_moves = [0 for _ in range(max_moves)]
    for c, i in enumerate(board.legal_moves):
        current_move = 0
        for j, item in enumerate(str(i)):
            if j % 2 == 0:
                try:
                    current_move += 10 ** j * files[item]
                except KeyError:
                    pass
                continue
            current_move += 10 ** j * int(item)
        possible_moves[c] = current_move
    brd = board_to_bitboard(board.fen())
    if board.turn == chess.WHITE: turn = 1
    else: turn = 0
    color = [turn]
    output = model.predict([color, [brd], [[possible_moves]]])
    #print(output, output.shape)
    #print(possible_moves)
    limit = possible_moves.index(0)
    #print('limit:', limit)
    output = output[0, :limit]
    #print(output)
    chosen_move_index = np.argmax(output)
    #print(chosen_move_index)
    pb = list(board.legal_moves)
    return (output, pb[chosen_move_index], possible_moves)


def auto_play(model, board, exploration_prob=0.2, verbose=True):
    X0 = []
    X1 = []
    Y = []
    if verbose: print(board)
    while (
        not board.is_checkmate() and not
        board.is_stalemate() and not
        board.is_insufficient_material() and not
        board.is_seventyfive_moves() and not
        board.is_repetition()):
        possible_moves = list(board.legal_moves)
        probability = np.random.rand()
        #print(type(probability))
        if probability < exploration_prob:
            l = len(possible_moves)
            output = np.zeros(shape=(l,))
            random_index = np.random.randint(l)
            output[random_index] = 1
            
            move = possible_moves[np.argmax(output)]
            pb = [0 for _ in range(max_moves)]
            for c, i in enumerate(board.legal_moves):
                current_move = 0
                for j, item in enumerate(str(i)):
                    if j % 2 == 0:
                        try:
                            current_move += 10 ** j * files[item]
                        except KeyError:
                            pass
                        continue
                    current_move += 10 ** j * int(item)
                pb[c] = current_move
            #print('Exploration')
        else:
            output, move, pb = make_move(model, board.fen())
        #print(move)
        #print(output)
        board.push(move)
        if verbose:
            print("===============")
            print(board)
        if output.shape != (max_moves,):
            temp = np.zeros(max_moves)
            temp[:len(output)] = output
            output = temp
        #print(output)
        X0.append(str(board.fen()))
        X1.append(pb)
        Y.append(output)

    if verbose:
        if board.is_checkmate():
            print("Checkmate!")
        elif board.is_stalemate():
            print("Stalemate")
        elif board.is_insufficient_material():
            print("Insufficient material")
        elif board.is_seventyfive_moves():
            print("Fifty-moves")
        elif board.is_repetition():
            print("Three-fold repetition")
        else:
            print("Game is still in progress.")
    result = board.result()
    if result == '1-0':
        who_won = 1
    elif result == '0-1':
        who_won = 0
    else:
        who_won = -1

    return who_won, X0, X1, Y


def train_model(model, fen, exploration_prob=0.2, play_iterations=200,
                training_verbose=True, playing_verbose=False, batch_size=None,
                shuffle=False, epochs=30):
    X0, X1, X2, Y = [], [], [], []

    for i in range(1, play_iterations + 1):
        if training_verbose: print(f'\rIteration: {i} / {play_iterations}', end='', flush=True)
        board = chess.Board(fen)
        who_won, X_c, X_c1, Y_c = auto_play(model,board, exploration_prob, playing_verbose)
        if (who_won == -1):
            continue
        if who_won == 1:
            X_c = X_c[0::2]
            X_c1 = X_c1[0::2]
            Y_c = Y_c[0::2]
        elif who_won == 0:
            X_c = X_c[1::2]
            X_c1 = X_c1[1::2]
            Y_c = Y_c[1::2]

        X0.append([who_won] * len(X_c))
        X1.append(X_c)
        X2.append(X_c1)
        Y.append(Y_c)
    if training_verbose: print()

    X0_concatenated = [item for sublist in X0 for item in sublist]
    X1_concatenated = [item for sublist in X1 for item in sublist]
    X2_concatenated = [item for sublist in X2 for item in sublist]

    for i, x in enumerate(X1_concatenated):
        X1_concatenated[i] = board_to_bitboard(x)

    X0 = np.array(X0_concatenated)
    X1 = np.array(X1_concatenated)
    X2 = np.array(list(map(np.array, X2_concatenated)))
    X2 = X2.reshape(X2.shape[0], 1, X2.shape[1])

    Y_concatenated = np.concatenate(Y, axis=0)
    
    model.fit((X0, X1, X2), y=Y_concatenated, batch_size=batch_size, shuffle=shuffle, epochs=epochs, verbose=training_verbose)


if __name__ == '__main__':
    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    model = new_model()
    train_model(model, fen, exploration_prob=0.5, batch_size=512, play_iterations=10)
    model.save("theseus.h5")
