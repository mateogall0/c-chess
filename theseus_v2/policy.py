#!/usr/bin/env python3
import torch
from stable_baselines3.ppo import MlpPolicy
from theseus_v2.wrappers import ChessWrapper2


class CustomPolicy(MlpPolicy):
    def forward(self, obs, deterministic=False):
        actions = super().forward(obs, deterministic)
        mask = self.get_action_mask(obs)
        masked_actions = actions * mask
        return masked_actions
    
    def get_action_mask(self, obs):
        for i in obs:
            b = ChessWrapper2.array_to_board(i)
            
        print(b)