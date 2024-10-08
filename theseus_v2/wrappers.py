#!/usr/bin/env python3
import numpy as np
from gym import spaces
import chess, gym, random, json
import chess.pgn
from typing import Tuple
from theseus_v2.config import DEBUG, input_shape, reward_factor, alphazero_action_space
from theseus_v2.board import decode_move


class ChessWrapper(gym.ObservationWrapper):
    """
    Chess environment wrapper.
    """

    def __init__(self, env, evaluator) -> None:
        super(ChessWrapper, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=1, shape=input_shape, dtype=np.float32)
        self.evaluator = evaluator
        self.reset()

    def update_action_space(self, restart=False) -> None:
        """
        Update the action space based on current board legal moves.
        """
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

    @property
    def board(self) -> chess.Board:
        return self.env._board

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
        
        return board

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
            reward = self.evaluator.evaluate_position(done, board_before, board_after, self.env, move)
        
        if DEBUG:
            print('(debug) move_uci:', move_uci)
            print('(debug) reward:', reward)
            print('(debug) done:', done)
            print('(debug) info:', info)
        
        if not done:
            self.update_action_space()

        if chose_illegal:
            reward = -2

        if info is None:
            info = {}

        return self.observation(obs), reward, done, info


    def reset(self, *ag,**kw):
        obs = self.env.reset()
        self.update_action_space()
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


class AlphaZeroChessWrapper(gym.Wrapper):
    def __init__(self, env, evaluator, initial_positions_path='data/initial_positions.json'):
        super(AlphaZeroChessWrapper, self).__init__(env)
        self.evaluator = evaluator
        with open(initial_positions_path, 'r') as f:
            self.initial_positions = json.load(f)
        self.actions = []

    def step(self, action):
        if action not in self.env.legal_actions:
            move = self.choose_legal_action(action)
        else:
            move = action

        self.actions.append(action)

        board_before = self.board.copy()
        board_after = board_before.copy()

        move_uci = self.env.decode(move)
        board_after.push(move_uci)

        obs, reward, done, info = self.env.step(move)
        if info is None:
            info = {}

        if self.evaluator is not None:
            evaluation_reward = self.evaluator.evaluate_position(
                done,
                board_before,
                board_after,
                self.env,
                move_uci
            )
            reward = evaluation_reward
        if DEBUG:
            print(f'(debug) reward: {reward} - done: {done} - info: {info}')
            print(f'(debug) action: {action} - chosen move: {move} - {move_uci}')
            print(f'(debug) before: {board_before.fen()}')
            print(f'(debug) average_action: {sum(self.actions) / len(self.actions)}')
            print('=' * 50)

        return obs, reward, done, info

    def observation(self, obs):
        return self.env.observation(obs)

    def choose_legal_action(self, action):
        """
        Fallback in case the provided action is not legal.
        This method chooses the closest legal action (or a random legal action if needed).
        """
        legal_actions = self.env.legal_actions
        closest_move = min(legal_actions, key=lambda num: abs(num - action))
        return closest_move

    @property
    def board(self) -> chess.Board:
        return self.env.get_board()

    def reset(self, fen=None):
        return self.env.reset()
        """
        self.env.reset()
        if fen is None:
            random_position = random.choice(list(self.initial_positions.keys()))
        else:
            random_position = fen
        if DEBUG:
            print(f'(debug) random position: {self.initial_positions[random_position]}')
        self.board.set_fen(random_position)

        return self.observation(self.board.copy())
        """

    def get_pgn(self) -> str:
        """
        Get an exportable Chess game string in PGN format.

        Returns:
            str: PGN string containing the whole Chess game.
        """
        board = self.board
        game = chess.pgn.Game.from_board(board)
        exporter = chess.pgn.StringExporter()
        pgn = game.accept(exporter)
        return pgn
    

class AlphaZeroWrapper2(gym.Wrapper):
    def step(self, action):
        if action not in self.env.legal_actions:
            print('Illegal action taken, stoping current episode.')
            return None, -0.01, True, {}
        else:
            move = action
        obs, reward, done, info = self.env.step(move)
        if info is None: info = {}
        if not done:
            obs, _, rdone, _ = self.env.step(random.choice(self.env.legal_actions)) # random move for black
            if rdone:
                reward = -1.0
                done = True
        return obs, reward, done, info
    
    def observation(cls, obs):
        return cls.env.observation(obs)

    @property
    def board(self):
        return self.env.get_board()
    
    def choose_legal_action(self, action):
        """
        Fallback in case the provided action is not legal.
        This method chooses the closest legal action (or a random legal action if needed).
        """
        legal_actions = self.env.legal_actions
        closest_move = min(legal_actions, key=lambda num: abs(num - action))
        return closest_move
    
    def get_pgn(self) -> str:
        """
        Get an exportable Chess game string in PGN format.

        Returns:
            str: PGN string containing the whole Chess game.
        """
        board = self.board
        game = chess.pgn.Game.from_board(board)
        exporter = chess.pgn.StringExporter()
        pgn = game.accept(exporter)
        return pgn
    
class ChessWrapper2(ChessWrapper):
    max_moves=200
    def update_action_space(self, restart=False) -> None:
        r = super().update_action_space(restart)
        self.action_space = spaces.Discrete(self.max_moves)
        return r

    def step(self, action):
        if DEBUG:
            print(f'(debug) action: {action} - action_space: {self.action_space} - len(legal_actions): {len(self.index_to_move)}')
        #move_uci = self.get_wrapped_index(self.index_to_move, int(action))
        try:
            move_uci = self.index_to_move[int(action)]
        except KeyError:
            print('Out of bounds move chosen')
            return None, -0.05, True, {}
        move = chess.Move.from_uci(move_uci)
        obs, reward, done, info = self.env.step(move)
        if info is None: info = {}
        #if self.evaluator:
        #    reward = self.evaluator.simple_evaluation(done, self.board)
        if not done:
            obs, _, rdone, _ = self.env.step(random.choice(list(self.board.legal_moves)))
            if rdone:
                reward = -1.0
            done = rdone
        self.update_action_space()
        return self.observation(obs), reward, done, info

    def get_wrapped_index(self, d, index):
        wrapped_index = index % len(d)
        return d[wrapped_index]
    
    def create_action_mask(self, legal_actions):
        legal_actions = list(legal_actions.keys())
        legal_actions = np.array(legal_actions, dtype=int)
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        mask[legal_actions] = 1
        
        return mask

class SyzygyWrapper(ChessWrapper2):
    max_moves = alphazero_action_space
    def __init__(self, env, evaluator, path='data/val_data/data.json') -> None:
        self.current_position_index = 0
        with open(path, 'r') as file:
            self.positions_expected = json.load(file)
        super().__init__(env, evaluator)
    
    def step(self, action: np.int64) -> Tuple[np.ndarray, float, bool, dict]:
        """
        """
        try:
            move = decode_move(int(action), self.board)
            obs, _, _, info = self.env.step(move)
        except Exception as e:
            print(f'Illegal action for Syzygy puzzle: {e}')
            return self.reset(), -0.1, True, {}
        if info is None: info = {}
        if str(move) == self.positions_expected[self.current_position_index][1]:
            reward = 1.0
        else:
            reward = -1.0
        self.update_action_space()
        return self.observation(obs), reward, True, info

    def reset(self):
        self.env.reset()
        self.env._board = chess.Board(self.positions_expected[self.current_position_index][0])
        self.update_action_space()
        self.current_position_index += 1
        if self.current_position_index >= len(self.positions_expected):
            self.current_position_index = 0
        return self.observation(self.board)


class TheseusChessWrapper(ChessWrapper2):
    """
    Final Chess wrapper implementation that uses the observation of
    `ChessWrapper` along with a move encode-decode functionality
    from the `ChessAlphaZero-v0` environment.

    Attributes:
        max_moves (int): AlphaZero action space, used as Discrete.
    """
    max_moves = alphazero_action_space

    def __init__(self,*ag,**kw) -> None:
        super().__init__(*ag,**kw)
        if self.evaluator is None:
            raise ValueError('evaluator must be a chess move evaluator.')

    @property
    def board(self) -> chess.Board:
        return self.env._board

    def step(self, action: np.int64, bot_only=True) -> Tuple[np.ndarray, float, bool, dict]:
        """
        This method takes an action, decodes it, and attempts to apply it to
        the current game state. If the action is valid, the environment is
        updated and a subsqeuent move is made using `make_move_external`.

        :Exceptions: If an illegal action is attempted, the environment is
            reset, a small punishment is given, and the episode is
            labeled as done.
        """
        try:
            move = decode_move(int(action), self.board)
            obs, reward, done, info = self.env.step(move)
        except Exception as e:
            print(f'Illegal action: {e}')
            return self.reset(), -0.1, True, {}
        if info is None: info = {}
        if not done and bot_only:
            # Move for black pieces
            obs, _, rdone, _ = self.env.step(self.make_move_external(self.board))
            if rdone and self.board.is_checkmate():
                reward = -1.0 # loss punishment
            done = rdone
        self.update_action_space()
        return self.observation(obs), reward, done, info

    def make_move_external(self, board: chess.Board, depth=1) -> chess.Move:
        """
        Make an external move from the evaluator, uses default engine given by
        the evaluator.
        """
        move = self.evaluator.make_move(board, depth=depth)
        return move
