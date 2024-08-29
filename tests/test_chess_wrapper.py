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
        ret_board = ChessWrapper.array_to_board(arr)
        self.assertIsInstance(ret_board, chess.Board)
        self.assertEqual(ret_board.fen(), fen)
