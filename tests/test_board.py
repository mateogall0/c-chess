#!/usr/bin/env python3
from unittest import TestCase
import gym, gym_chess
from theseus_v2.board import array_to_board
from theseus_v2.wrappers import AlphaZeroWrapper2


class TestBoard(TestCase):
    def test_rebuilt_board(self):
        env = AlphaZeroWrapper2(gym.make('ChessAlphaZero-v0'))
        obs = env.reset()
        b = env.board
        rebuilt = array_to_board(obs)
        self.assertEqual(b, rebuilt)