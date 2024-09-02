#!/usr/bin/env python3
import gym, json
import numpy as np
from gym import spaces
import chess, gym, random
import chess.pgn
from typing import Tuple
from theseus_v2.config import DEBUG, input_shape, reward_factor
from gym_chess.alphazero import BoardEncoding


class ChessWrapper(gym.ObservationWrapper):
    """
    Chess environment wrapper.
    """

    def __init__(self, env, evaluator) -> None:
        super(ChessWrapper, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=1, shape=input_shape, dtype=np.float32)
        self.evaluator = evaluator
        self.update_action_space()

    def update_action_space(self, restart=False) -> None:
        """
        Update the action space based on current board legal moves.
        """
        if self.env._board == None or restart:
            self.env.reset()
        self.move_to_index, self.index_to_move = self._create_action_space(self.env._board)
        self.action_space = spaces.Discrete(len(self.move_to_index))

    @classmethod
    def _create_action_space(cls, board: chess.Board) -> Tuple[dict, dict]:
        move_to_index = {}
        index_to_move = {}
        index = 0
        for move in board.legal_moves:
            move_to_index[move.uci()] = index
            index_to_move[index] = move.uci()
            index += 1
        return move_to_index, index_to_move

    def observation(self, obs) -> np.ndarray:
        self.arr = self.board_to_array(obs)
        return self.arr

    @classmethod
    def board_to_array(cls, board: chess.Board) -> np.ndarray:
        """
        Turns a board into a model-readable array.

        Args:
            board (chess.Board): Input board.

        Returns:
            np.ndarray: Processed array board.
        """
        piece_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        board_array = np.zeros(input_shape, dtype=np.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = chess.square_rank(square)
                col = chess.square_file(square)
                layer = piece_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
                board_array[row, col, layer] = 1

        if board.turn == chess.WHITE:
            board_array[:, :, 12] = 1
        
        if board.has_kingside_castling_rights(chess.WHITE):
            board_array[:, :, 13] = 1
        if board.has_queenside_castling_rights(chess.WHITE):
            board_array[:, :, 14] = 1
        if board.has_kingside_castling_rights(chess.BLACK):
            board_array[:, :, 15] = 1
        if board.has_queenside_castling_rights(chess.BLACK):
            board_array[:, :, 16] = 1
        
        possible_moves_layer = np.zeros((8, 8), dtype=np.float32)
        for move in board.legal_moves:
            from_square = move.from_square
            from_row = chess.square_rank(from_square)
            from_col = chess.square_file(from_square)
            possible_moves_layer[from_row, from_col] = 1
        
        board_array[:, :, 17] = possible_moves_layer

        return board_array
    
    @classmethod
    def array_to_board(cls, board_array: np.ndarray) -> chess.Board:
        """
        Turns a model-readable array back into a chess board.

        Args:
            board_array (np.ndarray): Processed array board.

        Returns:
            chess.Board: The corresponding chess board.
        """
        piece_map = {
            0: chess.PAWN,
            1: chess.KNIGHT,
            2: chess.BISHOP,
            3: chess.ROOK,
            4: chess.QUEEN,
            5: chess.KING
        }

        board = chess.Board()
        board.clear()

        for square in chess.SQUARES:
            row = chess.square_rank(square)
            col = chess.square_file(square)
            
            for layer in range(12):
                if board_array[row, col, layer] == 1:
                    piece_type = piece_map[layer % 6]
                    color = chess.BLACK if layer >= 6 else chess.WHITE
                    board.set_piece_at(square, chess.Piece(piece_type, color))

        board.turn = chess.WHITE if board_array[0, 0, 12] == 1 else chess.BLACK

        if board_array[0, 0, 13] == 1:
            board.castling_rights |= chess.BB_H1
        if board_array[0, 0, 14] == 1:
            board.castling_rights |= chess.BB_A1
        if board_array[0, 0, 15] == 1:
            board.castling_rights |= chess.BB_H8
        if board_array[0, 0, 16] == 1:
            board.castling_rights |= chess.BB_A8
        
        possible_moves = []
        for row in range(8):
            for col in range(8):
                if board_array[row, col, 17] == 1:
                    from_square = chess.square(col, row)
                    piece = board.piece_at(from_square)
                    if piece:
                        for move in board.legal_moves:
                            if move.from_square == from_square:
                                possible_moves.append(move.uci())
        return board, possible_moves

    @classmethod
    def legal_moves_to_array(cls, moves: dict, move_to_index: dict) -> np.ndarray:
        keys = []
        indices = []
        for k, v in moves.items():
            keys.append(k)
            indices.append(move_to_index[v])
        k_array = np.array(keys)
        indices_array = np.array(indices)
        structured_array = np.stack((k_array, indices_array), axis=1).T
        return structured_array


    def step(self, action: np.int64, playing=False) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Executes a step in the current game state.

        Args:
            action (np.int64): Action to be executed.
            playing (bool): If True, the game is in playing mode (as opposed to training).

        Returns:
            np.ndarray: Observation.
            float: Reward for the current action.
            bool: True if the game is finished.
            dict: Info dictionary.
        """
        if DEBUG: print('(debug) action:', action)

        _, legal_moves = self.array_to_board(self.arr)
        
        chose_illegal = False
        info = {}

        try:
            move_uci = legal_moves[int(action)]
        except IndexError:
            move_uci = random.choice(legal_moves)
            chose_illegal = True
            info['random_move'] = True
        
        if DEBUG:
            print(f'(debug) legal moves: {legal_moves}')
            print(f'(debug)\n{self.env._board}')
            print(f'(debug) action: {action} - move_uci: {move_uci}')
        
        board_before = self.env._board.copy()
        
        move = chess.Move.from_uci(move_uci)
        
        obs, reward, done, info = self.env.step(move)
        
        board_after = self.env._board.copy()
        
        if not chose_illegal and self.evaluator:
            reward += self.evaluator.evaluate_position(done, board_before, board_after, self.env, move)
        
        if DEBUG:
            print('(debug) move_uci:', move_uci)
            print('(debug) reward:', reward)
            print('(debug) done:', done)
            print('(debug) info:', info)
        
        if not done:
            self.update_action_space()

        if chose_illegal:
            reward = -10.0 / reward_factor

        if info is None:
            info = {}

        return self.observation(obs), reward, done, info


    def reset(self):
        obs = self.env.reset()
        self.update_action_space(restart=True)
        return self.observation(obs)

    def render(self, mode='unicode') -> None:
        """
        Render the environment.

        Args:
            mode (str): Mode used to render.
        """
        print(self.env.render(mode=mode))
        print('=' * 15)

    def get_pgn(self) -> str:
        """
        Get an exportable Chess game string.

        Returns:
            str: PGN string containing a whole exportable game of Chess.
        """
        board = self.env._board
        game = chess.pgn.Game.from_board(board)
        exporter = chess.pgn.StringExporter()
        pgn = game.accept(exporter)
        return pgn


class SyzygyWrapper(ChessWrapper):
    def __init__(self, env, evaluator, path='data/val_data/data.json') -> None:
        super().__init__(env, evaluator)
        self.current_position_index = 0
        with open(path, 'r') as file:
            self.positions_expected = json.load(file)
        self.update_action_space()
    
    def step(self, action: np.int64) -> Tuple[np.ndarray, float, bool, dict]:
        """
        """
        legal_moves = [str(move) for move in self.env._board.legal_moves]
        chose_ilegal = False
        try:
            move_uci = self.index_to_move[int(action)]
        except KeyError:
            move_uci = random.choice(legal_moves)
            chose_ilegal = True
            info = {'random_move': True}
        if DEBUG:
            print(f'(debug) Syzygy training -  action: {action} - index_to_move: {self.index_to_move} - move_uci : {move_uci}')
        move = chess.Move.from_uci(move_uci)
        try:
            move = chess.Move.from_uci(move_uci)
            obs, reward, done, info = self.env.step(move)
        except ValueError as e:
            if DEBUG:
                print(f'(debug) Illegal move exception: {e}')
            obs = self.observation(self.env._board)
            chose_ilegal = True
            done = False
            info = {'illegal_move': True}
        if move_uci == self.positions_expected[self.current_position_index][1]:
            reward = 10.0 / reward_factor
        self.current_position_index = (self.current_position_index + 1) % len(self.positions_expected)
        self.env._board = chess.Board(self.positions_expected[self.current_position_index][0])
        if chose_ilegal: reward = -10.0 / reward_factor
        if DEBUG:
            print('(debug) Syzygy training -', move_uci, reward, done, info)
        if info is None:
            info = {}
        self.update_action_space()

        return self.observation(self.env._board),  reward, done, info

class AlphaZeroChessWrapper(gym.Wrapper):
    def __init__(self, env, evaluator):
        super(AlphaZeroChessWrapper, self).__init__(env)
        self.evaluator = evaluator

    @classmethod
    def find_closest_move(cls, moves, move):
        closest_move = min(moves, key=lambda num: abs(num - move))
        distance = abs(closest_move - move)
        if DEBUG:
            print(f'(debug) moves: {moves} - move: {move} - closest: {closest_move} - distance: {distance}')
        return closest_move, distance

    def step(self, action):
        move, move_distance = self.find_closest_move(self.env.legal_actions, action)
        board_before = self.board.copy()
        board_after = board_before.copy()
        move_uci = self.env.decode(move)
        board_after.push(move_uci)

        obs, reward, done, info = self.env.step(move)
        if self.evaluator is not None:
            reward += self.evaluator.evaluate_position(done, board_before, board_after, self.env, move_uci)
        reward -= abs(move_distance)
        if info is None: info = {}
        if DEBUG:
            print(f'(debug) obs: {obs} - reward: {reward} - done: {done} - info: {info} - board: \n{board_after}')
        return obs, reward / reward_factor, done, info

    def observation(self, obs):
        return self.env.observation(obs)

    @property
    def board(self):
        return self.env.get_board()

    def reset(self):
        return self.env.reset()
    
    def get_pgn(self) -> str:
        """
        Get an exportable Chess game string.

        Returns:
            str: PGN string containing a whole exportable game of Chess.
        """
        board = self.board
        game = chess.pgn.Game.from_board(board)
        exporter = chess.pgn.StringExporter()
        pgn = game.accept(exporter)
        return pgn
