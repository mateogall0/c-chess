#!/usr/bin/env python3
import chess
import numpy as np
import chess.engine
from config import DEBUG
import asyncio


class Evaluator:

    external_paths = {
        'stockfish': 'bin/stockfish'
    }

    def __init__(self):
        self.external = {}
        for name, path in self.external_paths.items():
            self.external[name] = chess.engine.SimpleEngine.popen_uci(
                path
            )


    def evaluate_position(self, done: bool, board_before: chess.Board,
                          board_after: chess.Board, env, move_done: chess.Move) -> float:
        reward = 0.0
        if done:
            if board_after.is_checkmate():
                reward = 100.0 / len(env._board.move_stack)
            elif board_after.is_stalemate():
                reward = -2.0
            elif board_after.is_insufficient_material() or board_after.is_repetition() or board_after.can_claim_fifty_moves():
                reward = 0.0
        else:
            # Evaluate from both perspectives
            player_reward = self.evaluate_for_side(board_after, not board_after.turn)
            opponent_reward = self.evaluate_for_side(board_after, board_after.turn)
            
            # Reward for capturing a piece
            if len(board_before.piece_map()) > len(board_after.piece_map()):
                player_reward += 0.2  # Reward for capturing a piece
                
            # Combine player and opponent evaluations
            reward += player_reward - opponent_reward
            loop = asyncio.get_event_loop()
            reward += loop.run_until_complete(
                self.external_evaluation(board_before, move_done)
            )

        return reward

    def evaluate_for_side(self, board: chess.Board, side: bool) -> float:
        reward = 0.0
        # Evaluate piece activity
        #reward += self.piece_activity(board, side)
        
        # Evaluate pawn structure
        reward += self.pawn_structure(board, side)
        
        # Evaluate king safety
        reward += self.king_safety(board, side)
        
        # Evaluate control of the center
        reward += self.center_control(board, side)
        
        return reward

    def pawn_structure(self, board: chess.Board, side: bool) -> float:
        structure_score = 0.0
        pawns = [square for square, piece in board.piece_map().items() if piece.piece_type == chess.PAWN and piece.color == side]
        for square in pawns:
            if self.is_isolated(board, square, side):
                structure_score -= 0.1  # Penalize isolated pawns
            if self.is_doubled(board, square, side):
                structure_score -= 0.1  # Penalize doubled pawns
        return structure_score

    def is_isolated(self, board: chess.Board, square: chess.Square, side: bool) -> bool:
        file_index = chess.square_file(square)
        adjacent_files = [file_index - 1, file_index + 1]
        for adj_file in adjacent_files:
            if 0 <= adj_file <= 7:
                for rank in range(8):
                    piece = board.piece_at(chess.square(adj_file, rank))
                    if piece and piece.piece_type == chess.PAWN and piece.color == side:
                        return False
        return True

    def is_doubled(self, board: chess.Board, square: chess.Square, side: bool) -> bool:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        for rank in range(8):
            if rank != rank_index:
                piece = board.piece_at(chess.square(file_index, rank))
                if piece and piece.piece_type == chess.PAWN and piece.color == side:
                    return True
        return False

    def piece_activity(self, board: chess.Board, side: bool) -> float:
        activity_score = 0.0
        # Evaluate activity of pieces (e.g., active piece positions)
        for square, piece in board.piece_map().items():
            if piece.color == side and piece.piece_type != chess.PAWN:
                # Use square directly from piece_map
                if self.is_active_position(board, square):
                    activity_score += 0.1
        return activity_score

    def is_active_position(self, board: chess.Board, square: chess.Square) -> bool:
        # TODO
        return True

    def king_safety(self, board: chess.Board, side: bool) -> float:
        safety_score = 0.0
        king_square = board.king(side)
        if king_square:
            # Check for surrounding pawns and pieces
            if board.is_attacked_by(not side, king_square):
                safety_score -= 0.2
        return safety_score

    def center_control(self, board: chess.Board, side: bool) -> float:
        control_score = 0.0
        center_squares = [chess.square(3, 3), chess.square(3, 4), chess.square(4, 3), chess.square(4, 4)]
        for square in center_squares:
            piece = board.piece_at(square)
            if piece and piece.color == side:
                control_score += 0.1
        return control_score

    def __del__(self):
        for k in self.external.keys():
            self.external[k].close()

    async def external_evaluation(self, board_before: chess.Board, move_done: chess.Move) -> float:
        rewards = 0.0
        tasks = [
            self.external_move(engine, board_before)
            for engine in self.external.values()
        ]
        external_moves = await asyncio.gather(*tasks)
        if DEBUG:
            print('(debug)', external_moves)
        for external_move in external_moves:
            if external_move == move_done:
                rewards += 2
            else:
                rewards -= 0.1
        return rewards

    async def external_move(self, engine, board):
        external_move = engine.play(board, chess.engine.Limit(time=0.1))
        return external_move
