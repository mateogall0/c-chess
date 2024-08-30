#!/usr/bin/env python3
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMlpPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMlpPolicy, self).__init__(*args, **kwargs)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-5, weight_decay=1e-5)

    def _build_mlp(self, net_arch, activation_fn):
        return nn.Sequential(
            nn.Linear(self.features_dim, net_arch[0]),
            activation_fn(),
            nn.Linear(net_arch[0], net_arch[1]),
            activation_fn(),
            nn.Linear(net_arch[1], self.action_dim)
        )
