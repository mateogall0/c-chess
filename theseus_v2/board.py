#!/usr/bin/env python3
import numpy as np
import chess
from gym_chess.alphazero.move_encoding import queenmoves, knightmoves, underpromotions

def encode_move(move: chess.Move) -> int:
    action = queenmoves.encode(move)

    if action is None:
        action = knightmoves.encode(move)

    if action is None:
        action = underpromotions.encode(move)

    if action is None:
        raise ValueError(f"{move} is not a valid move")

    return action

def decode_move(action: int, board: chess.Board) -> chess.Move:
    turn = chess.WHITE
    move = queenmoves.decode(action)
    is_queen_move = move is not None

    if not move:
        move = knightmoves.decode(action)

    if not move:
        move = underpromotions.decode(action)

    if not move:
        raise ValueError(f"{action} is not a valid action")

    if is_queen_move:
        to_rank = chess.square_rank(move.to_square)
        is_promoting_move = (
            (to_rank == 7 and turn == chess.WHITE) or 
            (to_rank == 0 and turn == chess.BLACK)
        )


        piece = board.piece_at(move.from_square)
        is_pawn = piece.piece_type == chess.PAWN

        if is_pawn and is_promoting_move:
            move.promotion = chess.QUEEN

    return move
