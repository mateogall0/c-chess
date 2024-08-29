#!/usr/bin/env python3
from unittest import TestCase
import chess
from theseus_v2 import Evaluator, config


class TestEvaluator(TestCase):
    def test_external_stockfish_d20_positive(self):
        config.EXTERNAL_EVALUATION_DEPTH_LIMIT=20
        ev = Evaluator()
        fen_before = 'b1Nk4/p4r1Q/5p2/1p2p3/n3Pq2/b5PK/PPPP1P1P/R1BBN2R b - - 2 45'
        fen_after = 'b1Nk4/p6r/5p2/1p2p3/n3Pq2/b5PK/PPPP1P1P/R1BBN2R w - - 0 46'
        board_before = chess.Board(fen=fen_before)
        board_after = chess.Board(fen=fen_after)
        reward = ev.get_external_reward(ev.external['stockfish'], board_before, board_after)
        self.assertEqual(reward, 3.51)

    def test_external_stockfish_d15_positive(self):
        config.EXTERNAL_EVALUATION_DEPTH_LIMIT=15
        ev = Evaluator()
        fen_before = 'b1Nk4/p4r1Q/5p2/1p2p3/n3Pq2/b5PK/PPPP1P1P/R1BBN2R b - - 2 45'
        fen_after = 'b1Nk4/p6r/5p2/1p2p3/n3Pq2/b5PK/PPPP1P1P/R1BBN2R w - - 0 46'
        board_before = chess.Board(fen=fen_before)
        board_after = chess.Board(fen=fen_after)
        reward = ev.get_external_reward(ev.external['stockfish'], board_before, board_after)
        self.assertEqual(reward, 3.51)

    def test_external_stockfish_d1_positive(self):
        config.EXTERNAL_EVALUATION_DEPTH_LIMIT=1
        ev = Evaluator()
        fen_before = 'b1Nk4/p4r1Q/5p2/1p2p3/n3Pq2/b5PK/PPPP1P1P/R1BBN2R b - - 2 45'
        fen_after = 'b1Nk4/p6r/5p2/1p2p3/n3Pq2/b5PK/PPPP1P1P/R1BBN2R w - - 0 46'
        board_before = chess.Board(fen=fen_before)
        board_after = chess.Board(fen=fen_after)
        reward = ev.get_external_reward(ev.external['stockfish'], board_before, board_after)
        self.assertEqual(reward, 3.51)

    def test_external_stockfish_d20_negative(self):
        config.EXTERNAL_EVALUATION_DEPTH_LIMIT=20
        ev = Evaluator()
        fen_before = 'b1Nk4/p6r/5p2/1p2p3/n3Pq2/b5P1/PPPP1PKP/R1BBN2R b - - 1 46'
        fen_after = 'b1Nk4/p6r/5p2/1p2p3/nb2Pq2/6P1/PPPP1PKP/R1BBN2R w - - 2 47'
        board_before = chess.Board(fen=fen_before)
        board_after = chess.Board(fen=fen_after)
        reward = ev.get_external_reward(ev.external['stockfish'], board_before, board_after)
        self.assertEqual(reward, -5.33)
