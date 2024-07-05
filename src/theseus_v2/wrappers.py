#!/usr/bin/env python3
import gym
import numpy as np
from gym import spaces
import chess, gym, random

class ChessWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ChessWrapper, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)
        self.move_to_index, self.index_to_move = self._create_action_space()
        self.action_space = spaces.Discrete(len(self.move_to_index))

    def _create_action_space(self):
        move_to_index = {}
        index_to_move = {}
        index = 0
        for from_square in chess.SQUARES:
            for to_square in chess.SQUARES:
                for promotion in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    move = chess.Move(from_square, to_square, promotion=promotion)
                    if chess.Board().is_legal(move):
                        move_to_index[move.uci()] = index
                        index_to_move[index] = move.uci()
                        index += 1
        return move_to_index, index_to_move

    def observation(self, obs):
        return self.board_to_array(obs)

    def board_to_array(self, board):
        piece_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        board_array = np.zeros((8, 8, 12), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                piece_type = piece_map[piece.piece_type]
                color_offset = 6 if piece.color == chess.BLACK else 0
                board_array[row, col, piece_type + color_offset] = 1
        return board_array

    def step(self, action):
        move_uci = self.index_to_move[action]
        move = chess.Move.from_uci(move_uci)
        if not self.env._board.is_legal(move):
            legal_moves = [move for move in self.env._board.legal_moves]
            move = random.choice(legal_moves)
            info = {'random_move': True}

        obs, reward, done, info = self.env.step(move)
        if info is None:
            info = {}
        return self.observation(obs), reward, done, info
    
    def render(self,):
        print(self.env.render(mode='unicode'))