#!/usr/bin/env python3
"""
This module contains the Theseus Bot class.

About the modules imported:
- Keras: used for loading previously trained models.
- Python-Chess: brings management for the game of Chess.
- NumPy: manages large arrays and makes random choices.
- Matplotlib: used to plot data obtained from the bot's traning.
"""
from tensorflow import keras as K
import chess
import numpy as np
import matplotlib.pyplot as plt

try: from .model import layers
except ImportError: from model import layers

class Bot:
    max_moves = 128
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
    chess_openings = [
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', # standard starting position
        'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2', # Sicilian Defense
        'rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPP1P/RNBQKBNR b KQkq - 0 1', # King's Fianchetto Opening
        'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1', # Queen's Pawn Opening
        'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1', # King's Pawn Opening
        'rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1', #Nimzowitsch-Larsen Attack
        # TO DO
    ]
    __predefined_probability_of_exploration = [
        1,
        0.05,
    ]

    def __init__(self, new_model=False, path='theseus.h5'):
        if new_model:
            self.__engine = self.new_model()
            self.__is_new = True
        else:
            self.__is_new = False
            self.__engine = K.models.load_model(path)
        self.__training_records = []

    @property
    def is_new(self):
        return self.__is_new
    
    @property
    def engine(self):
        return self.__engine

    @property
    def engine_summary(self):
        return self.__engine.summary()

    @property
    def training_records(self):
        return self.__training_records

    def engine_save(self, path='theseus.h5'):
        return self.__engine.save(path)

    def new_model(self):
        (merged_inputs, color_input_layer, board_input_layer,
         moves_input_layer) = layers.input_layers(self.max_moves)

        output_layer = layers.hidden_layers(self.max_moves, merged_inputs)
        return layers.compile_model(
            output_layer,
            color_input_layer,
            board_input_layer,
            moves_input_layer
        )

    def default_session_train(self):
        return self.session_train_model(
            batch_size=256,
            play_iterations=4096, epochs=1024,
            training_iterations=2
        )

    @staticmethod
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

    def train_model(self, exploration_prob=0.2, play_iterations=200,
            training_verbose=True, playing_verbose=False, batch_size=32,
            shuffle=True, epochs=30, keras_verbose=False, validation_data=()):

        X0, X1, X2, Y = [], [], [], []
        fen_codes_lenght = len(self.chess_openings)

        for i in range(1, play_iterations + 1):
            if training_verbose: print(f'\r  Playing iteration: {i} / {play_iterations}', end='', flush=True)
            try: chess_opening_index = (i) % fen_codes_lenght
            except ZeroDivisionError: chess_opening_index = 0
            board = chess.Board(self.chess_openings[chess_opening_index])
            who_won, X_c, X_c1, Y_c = self.__auto_play(self.__engine, board, exploration_prob, playing_verbose)
            if (who_won == -1):
                continue
            y = 0 if who_won == 1 else 1
            X_c = X_c[y::2]
            X_c1 = X_c1[y::2]
            Y_c = Y_c[y::2]

            X0.append([who_won] * len(X_c))
            X1.append(X_c)
            X2.append(X_c1)
            Y.append(Y_c)
        if training_verbose: print()

        X0_concatenated = [item for sublist in X0 for item in sublist]
        X1_concatenated = [item for sublist in X1 for item in sublist]
        X2_concatenated = [item for sublist in X2 for item in sublist]

        for idx, x in enumerate(X1_concatenated):
            X1_concatenated[idx] = self.board_to_bitboard(x)

        X0 = np.array(X0_concatenated)
        X1 = np.array(X1_concatenated)
        X2 = np.array(list(map(np.array, X2_concatenated)))
        try:
            X2 = X2.reshape(X2.shape[0], 1, X2.shape[1])
        except IndexError:
            print(f'In none of the games is a single win.')
            return

        Y_concatenated = np.concatenate(Y, axis=0)

        return self.__engine.fit((X0, X1, X2), y=Y_concatenated, batch_size=batch_size,
                         shuffle=shuffle, epochs=epochs, verbose=keras_verbose,
                         validation_data=validation_data)

    def __auto_play(self, model, board, exploration_prob=0.2, verbose=True):
        X0 = []
        X1 = []
        Y = []
        while (
            not board.is_checkmate() and not
            board.is_stalemate() and not
            board.is_insufficient_material() and not
            board.is_seventyfive_moves() and not
            board.is_repetition()):
            possible_moves = list(board.legal_moves)
            probability = np.random.rand()
            if probability < exploration_prob:
                l = len(possible_moves)
                output = np.zeros(shape=(l,))
                random_index = np.random.randint(l)
                output[random_index] = 1
                
                move = possible_moves[np.argmax(output)]
                pb = [0 for _ in range(self.max_moves)]
                for c, i in enumerate(board.legal_moves):
                    current_move = 0
                    for j, item in enumerate(str(i)):
                        if j % 2 == 0:
                            try:
                                current_move += 10 ** j * self.files[item]
                            except KeyError:
                                pass
                            continue
                        current_move += 10 ** j * int(item)
                    pb[c] = current_move
            else:
                output, move, pb = self.make_move(model, board.fen())
            board.push(move)
            """if verbose:
                print("===============")
                print(board)"""
            if output.shape != (self.max_moves,):
                temp = np.zeros(self.max_moves)
                temp[:len(output)] = output
                output = temp
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

    def session_train_model(self,
                            play_iterations=200,
                            training_verbose=True, playing_verbose=False, batch_size=None,
                            shuffle=True, epochs=30,
                            training_iterations=5, keras_verbose=False):
        self.__training_records = []
        val_data = np.load('../../data/val_data/syzygy.npz')
        X0_val = val_data['X0']
        X1_val = val_data['X1']
        X2_val = val_data['X2'].reshape(695, 1, 128)
        Y_val = val_data['Y']
        switch_exploration_prob_at = training_iterations // len(self.__predefined_probability_of_exploration)
        i_exploration_prob = 0

        for i in range(training_iterations):
            if i != 0 and i % switch_exploration_prob_at == 0: i_exploration_prob += 1
            try:
                current_exploration_prob = self.__predefined_probability_of_exploration[
                    i_exploration_prob
                ]
            except IndexError:
                i_exploration_prob = len(self.__predefined_probability_of_exploration) - 1
                current_exploration_prob = self.__predefined_probability_of_exploration[
                    i_exploration_prob
                ]
            print(f'Training session: {i + 1} / {training_iterations} at {current_exploration_prob * 100}% probability of a random move')
            history = self.train_model(exploration_prob=current_exploration_prob,
                        batch_size=batch_size, play_iterations=play_iterations, epochs=epochs,
                        playing_verbose=playing_verbose, training_verbose=training_verbose,
                        shuffle=shuffle, keras_verbose=keras_verbose, validation_data=([X0_val, X1_val, X2_val], Y_val))
            self.__training_records.append(history)
            if history is None:
                print('  Nothing to learn from these iterations...')
            elif not keras_verbose:
                print(f'  Last recorded validation accuracy: {history.history["val_acc"][-1]}')
                print(f'  Last recorded accuracy: {history.history["acc"][-1]}')
        return self.__training_records

    def make_move(self, model, fen):
        board = chess.Board(fen)
        possible_moves = [0 for _ in range(self.max_moves)]
        for c, i in enumerate(board.legal_moves):
            current_move = 0
            for j, item in enumerate(str(i)):
                if j % 2 == 0:
                    try:
                        current_move += 10 ** j * self.files[item]
                    except KeyError:
                        pass
                    continue
                current_move += 10 ** j * int(item)
            possible_moves[c] = current_move
        brd = self.board_to_bitboard(board.fen())
        if board.turn == chess.WHITE: turn = 1
        else: turn = 0
        color = [turn]
        output = model.predict([color, [brd], [[possible_moves]]])
        limit = possible_moves.index(0)
        output = output[0, :limit]
        chosen_move_index = np.argmax(output)
        pb = list(board.legal_moves)
        return (output, pb[chosen_move_index], possible_moves)

    def plot_training_records(self):
        legend_labels = []
        for i, history in enumerate(self.training_records):
            if history is None: continue
            plt.plot(history.history['loss'], label=f'Training Run {i + 1}')
            plt.plot(history.history['val_loss'], label=f'Validation Run {i + 1}')
            legend_labels.append(f'Training Run {i + 1}')
            legend_labels.append(f'Validation Run {i + 1}')

        legend_labels.sort()

        plt.title('Training and Validation Losses for Multiple Runs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(legend_labels)
        plt.show()

    def play(self,
             engine_only=False,
             fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
             human_moves_at=None):
        if human_moves_at != 'white' and human_moves_at != 'black':
            human_moves_at = np.random.choice(['white', 'black'])
        print(human_moves_at)
        board = chess.Board(fen)
        turn = 'black'
        while not board.is_game_over():
            turn = 'black' if turn == 'white' else 'white'
            print(board)

            if turn == human_moves_at and not engine_only:
                while True:
                    move_str = input(f'{turn.capitalize()} move: ')
                    try:
                        move = chess.Move.from_uci(move_str)
                        break
                    except ValueError:
                        print('Invalid input, try again.')
            else:
                _, move, _ = self.make_move(self.__engine, board.fen())
            board.push(move)
            print()

        if board.result() == "1/2-1/2": print('Draw.')
        else: print(f'{turn.capitalize()} wins!')

if __name__ == '__main__':
    """
    When executing this module a new bot will start its
    default training session.

    This is used for testing and demonstration purposes.
    """
    test = Bot(new_model=True)

    test.default_session_train()

    test.engine_save()
    test.plot_training_records()
