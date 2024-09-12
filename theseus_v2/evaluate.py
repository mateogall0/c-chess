#!/usr/bin/env python3
import chess
import chess.engine
from theseus_v2.config import DEBUG, EXTERNAL_EVALUATION_DEPTH_LIMIT, reward_factor
import asyncio


class Evaluator:
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 11 #theoretical value of king
    }
    external_paths = {
        'stockfish': 'bin/stockfish'
    }

    def __init__(self):
        self.external = {}
        for name, path in self.external_paths.items():
            self.external[name] = chess.engine.SimpleEngine.popen_uci(
                path
            )
            self.external[name].configure({"Threads": 1})


    def evaluate_position(self, done: bool, board_before: chess.Board,
                          board_after: chess.Board, env, move_done: chess.Move) -> float:
        reward = 0.0
        if not done:
            #reward += self.evaluate_pieces(board_after)
            #reward += self.evaluate_capture(board_before, move_done)
            #reward = reward / 2
            #reward += self.center_control(board_after)
            loop = asyncio.get_event_loop()
            reward += loop.run_until_complete(
                self.external_evaluation(board_before, board_after)
            )
        else: reward = 1.0
        return reward
    
    def evaluate_pieces(self, board: chess.Board):
        white_value = sum(
        self.get_piece_value(board.piece_at(square))
        for square in board.piece_map()
            if board.piece_at(square).color == chess.WHITE and board.piece_at(square).piece_type != chess.KING
        )
        black_value = sum(
            self.get_piece_value(board.piece_at(square))
            for square in board.piece_map()
            if board.piece_at(square).color == chess.BLACK and board.piece_at(square).piece_type != chess.KING
        )
        
        if board.turn != chess.WHITE:
            value_difference = white_value - black_value
        else:
            value_difference = black_value - white_value

        max_difference = white_value + black_value
        max_difference = max(max_difference, 1)

        normalized_reward = value_difference / max_difference
        normalized_reward = max(-1, min(1, normalized_reward))
        return normalized_reward

    def get_piece_value(self, piece: chess.Piece) -> float:
        return self.piece_values.get(piece.piece_type, 0)
    
    def evaluate_capture(self, board_before: chess.Board, last_move: chess.Move) -> float:
        """
        Possible maximum reward: 0.9
        """
        captured_pieces_value = 0.0

        captured_piece = board_before.piece_at(last_move.to_square)

        if captured_piece and captured_piece.piece_type != chess.KING:
            captured_piece_value = self.get_piece_value(captured_piece)
            captured_pieces_value += captured_piece_value

        return 0.1 * captured_pieces_value if captured_pieces_value > 0 else 0.0

    def get_piece_value(self, piece):
        if piece:
            return self.piece_values.get(piece.piece_type, 0)
        return 0

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

    def center_control(self, board_after: chess.Board) -> float:
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        extended_center_squares = [
            chess.C3, chess.C4, chess.C5, chess.C6,
            chess.F3, chess.F4, chess.F5, chess.F6
        ]

        center_control = 0.0
        for square in center_squares:
            if board_after.is_attacked_by(not board_after.turn, square):
                center_control += 0.1
            elif board_after.is_attacked_by(board_after.turn, square):
                center_control -= 0.1

        for square in extended_center_squares:
            if board_after.is_attacked_by(not board_after.turn, square):
                center_control += 0.05
            elif board_after.is_attacked_by(board_after.turn, square):
                center_control -= 0.05

        return center_control

    def __del__(self):
        for k in self.external.keys():
            self.external[k].close()

    async def external_evaluation(self, board_before: chess.Board, board_after: chess.Board) -> float:
        tasks = [
            self.get_external_reward_wrapper(engine, board_before, board_after, board_before.turn)
            for engine in self.external.values()
        ]
        external_evaluations = await asyncio.gather(*tasks)
        if DEBUG:
            print('(debug)', external_evaluations)
        rewards = sum(external_evaluations) / len(self.external)
        return rewards

    async def get_external_reward_wrapper(self, *ag, **kw) -> float:
        return self.get_external_reward(*ag, **kw)

    def get_external_reward(self, engine, board_before, board_after, turn) -> float:
        info_before = engine.analyse(board_before, chess.engine.Limit(depth=EXTERNAL_EVALUATION_DEPTH_LIMIT))
        score_before = info_before['score'].relative.score()

        info_after = engine.analyse(board_after, chess.engine.Limit(depth=EXTERNAL_EVALUATION_DEPTH_LIMIT))
        score_after = info_after['score'].relative.score()
        try:
            reward = score_before - score_after
            if turn == chess.BLACK:
                reward = reward * -1
            if len(board_before.move_stack()) < 10: # larger rewards and punishments for openings
                reward *= 2
        except:
            reward = 0.0
        reward = max(min(reward, reward_factor), -reward_factor)
        return reward / reward_factor
