#!/usr/bin/env python3
import chess
import numpy as np


class Evaluator:
    def evaulate_position(self, obs: np.ndarray,
                          done: bool, board_before: chess.Board,
                          board_after: chess.Board) -> float:
        reward = 0
        if done:
            if obs.is_checkmate():
                reward = 100.0 / len(self.env._board.move_stack)
            if obs.is_stalemate():
                reward = -2.0
            if obs.is_insufficient_material() or obs.is_repetition() or obs.can_claim_fifty_moves():
                reward = 0.0
        else:
            if len(board_before.piece_map()) > len(board_after.piece_map()):
                reward = 0.2
            else:
                reward = 0.0
        return reward