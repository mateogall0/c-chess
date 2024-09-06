#!/usr/bin/env python3
from unittest import TestCase
import chess
from theseus_v2 import Evaluator, config


class TestEvaluator(TestCase):
    def test_capture(self):
        ev = Evaluator()
        bf = chess.Board('rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2')
        m = chess.Move.from_uci('e4d5')
        r = ev.evaluate_capture(bf, m)
        self.assertEqual(r, 0.1)
        bf = chess.Board('rnbqkbnr/ppp1p1pp/8/3p1p2/4P1Q1/7P/PPPP1PP1/RNB1KBNR b KQkq - 0 3')
        m = chess.Move.from_uci('f5g4')
        r = ev.evaluate_capture(bf, m)
        self.assertEqual(r, 0.9)

    def test_no_capture(self):
        ev = Evaluator()
        bf = chess.Board('rnbqkbnr/ppp2ppp/4p3/3P4/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3')
        m = chess.Move.from_uci('c2c4')
        r = ev.evaluate_capture(bf, m)
        self.assertEqual(r, 0.0)

    def test_pieces_evaluation(self):
        ev = Evaluator()
        ba = chess.Board()
        r = ev.evaluate_pieces(ba)
        self.assertEqual(r, 0.0)
        
    def test_pieces_evaluation_negative(self):
        ev = Evaluator()
        ba = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPP1P/RNBQKBNR b KQkq - 0 1')
        r = ev.evaluate_pieces(ba)
        self.assertEqual(r, -0.012987012987012988)
        ba = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1')
        r = ev.evaluate_pieces(ba)
        self.assertEqual(r, -0.13043478260869565)
        ba = chess.Board('8/5q2/8/2k3K1/8/8/8/8 b - - 0 1')
        r = ev.evaluate_pieces(ba)
        self.assertEqual(r, -1.0)

    def test_pieces_evaluation_positive(self):
        ev = Evaluator()
        ba = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPP1P/RNBQKBNR w KQkq - 0 1')
        r = ev.evaluate_pieces(ba)
        self.assertEqual(r, 0.012987012987012988)
        ba = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1')
        r = ev.evaluate_pieces(ba)
        self.assertEqual(r, 0.13043478260869565)
        ba = chess.Board('8/5q2/8/2k3K1/8/8/8/8 w - - 0 1')
        r = ev.evaluate_pieces(ba)
        self.assertEqual(r, 1.0)
