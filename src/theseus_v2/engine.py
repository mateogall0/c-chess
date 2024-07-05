#!/usr/bin/env python3
import gym
import gym_chess
import chess


class Engine():
    
    def __init__(self,):
        self.env = gym.make('Chess-v0')


    def train(self):
        state = self.env.reset()

if __name__ == '__main__':
    engine = Engine()
    print(engine.env.render())