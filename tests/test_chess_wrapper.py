#!/usr/bin/env python3
from unittest import TestCase
from theseus_v2 import ChessWrapper
import chess
import numpy as np


class TestChessWrapper(TestCase):
    def test_board_to_array(self):
        fen = 'rnbqk2r/ppp2ppp/4pn2/3p4/1bPP4/2N5/PP1BPPPP/R2QKBNR w KQkq - 0 1'
        board = chess.Board(fen=fen)
        arr = ChessWrapper.board_to_array(board)
        self.assertIsInstance(arr, np.ndarray)
        ret_board = ChessWrapper.array_to_board(arr)[0]
        self.assertIsInstance(ret_board, chess.Board)
        self.assertEqual(ret_board.fen(), fen)

    def test_board_to_array_moves(self):
        fen = 'rnbqk2r/ppp2ppp/4pn2/3p4/1bPP4/2N5/PP1BPPPP/R2QKBNR w KQkq - 0 1'
        board = chess.Board(fen=fen)
        arr = ChessWrapper.board_to_array(board)
        moves = ChessWrapper.array_to_board(arr)[1]
        moves_from_board = []
        for m in board.legal_moves:
            moves_from_board.append(m.uci())
        self.assertEqual(sorted(moves), sorted(moves_from_board))

    def test_board_to_array_moves_with_indices(self):
        fen = 'rnbqk2r/ppp2ppp/4pn2/3p4/1bPP4/2N5/PP1BPPPP/R2QKBNR w KQkq - 0 1'
        board = chess.Board(fen=fen)
        arr = ChessWrapper.board_to_array(board)
        moves = ChessWrapper.array_to_board(arr)[1]
        _, index_to_move = ChessWrapper._create_action_space(board)
        for i, m in enumerate(moves):
            self.assertEqual(m, index_to_move[i])