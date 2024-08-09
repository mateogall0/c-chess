#!/usr/bin/env python3
import chess
import numpy as np

class Evaluator:
    def evaluate_position(self, obs: np.ndarray, done: bool, board_before: chess.Board,
                          board_after: chess.Board, env) -> float:
        reward = 0.0
        if done:
            if board_after.is_checkmate():
                reward = 100.0 / len(env._board.move_stack)
            elif board_after.is_stalemate():
                reward = -2.0
            elif board_after.is_insufficient_material() or board_after.is_repetition() or board_after.can_claim_fifty_moves():
                reward = 0.0
        else:
            # Reward for capturing a piece
            if len(board_before.piece_map()) > len(board_after.piece_map()):
                reward += 0.2
            
            # Evaluate the pawn structure after the move
            reward += self.pawn_structure(board_after)

        return reward
    
    def pawn_structure(self, board: chess.Board) -> float:
        structure_score = 0.0
        prior_turn = not board.turn
        pawns = [square for square, piece in board.piece_map().items() if piece.piece_type == chess.PAWN]
        for square in pawns:
            if self.is_isolated(board, square, prior_turn):
                structure_score -= 0.1  # Penalize isolated pawns
            if self.is_doubled(board, square, prior_turn):
                structure_score -= 0.1  # Penalize doubled pawns
        return structure_score

    def is_isolated(self, board: chess.Board, square: chess.Square, turn: bool) -> bool:
        file_index = chess.square_file(square)
        adjacent_files = [file_index - 1, file_index + 1]
        for adj_file in adjacent_files:
            if 0 <= adj_file <= 7:
                for rank in range(8):
                    piece = board.piece_at(chess.square(adj_file, rank))
                    if piece and piece.piece_type == chess.PAWN and piece.color == turn:
                        return False
        return True

    def is_doubled(self, board: chess.Board, square: chess.Square, turn: bool) -> bool:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        for rank in range(8):
            if rank != rank_index:
                piece = board.piece_at(chess.square(file_index, rank))
                if piece and piece.piece_type == chess.PAWN and piece.color == turn:
                    return True
        return False