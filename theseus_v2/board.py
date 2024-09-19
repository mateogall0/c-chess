#!/usr/bin/env python3
import numpy as np
import chess
def array_to_board(observation: np.array) -> chess.Board:
    meta = observation[:, :, -7:]

    turn = bool(meta[0, 0, 0])
    fullmove_number = int(meta[0, 0, 1])
    white_kingside_castle = bool(meta[0, 0, 2])
    white_queenside_castle = bool(meta[0, 0, 3])
    black_kingside_castle = bool(meta[0, 0, 4])
    black_queenside_castle = bool(meta[0, 0, 5])
    halfmove_clock = int(meta[0, 0, 6])

    board_representation = observation[:, :, :-7]

    board = chess.Board(fen=None)

    for row in range(8):
        for col in range(8):
            piece = board_representation[row, col]
            if piece.any():
                piece_type = np.argmax(piece[:6]) + 1
                color = bool(piece[6])
                board.set_piece_at(chess.square(col, row), chess.Piece(piece_type, color))

    board.turn = turn
    board.fullmove_number = fullmove_number
    board.halfmove_clock = halfmove_clock

    castling_rights = 0
    if white_kingside_castle:
        castling_rights |= chess.BB_H1
    if white_queenside_castle:
        castling_rights |= chess.BB_A1
    if black_kingside_castle:
        castling_rights |= chess.BB_H8
    if black_queenside_castle:
        castling_rights |= chess.BB_A8

    board.castling_rights = castling_rights

    return board