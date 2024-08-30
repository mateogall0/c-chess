#!/usr/bin/env python3
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


class CustomMlpPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU, ortho_init=True, *args, **kwargs):
        super(CustomMlpPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, ortho_init, *args, **kwargs)

        self.fc1 = nn.Linear(8 * 8 * 18, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_pi = nn.Linear(256, self.action_space.n)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, obs):
        obs = obs.float().reshape(obs.size(0), -1)

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        pi = self.fc_pi(x)
        value = self.fc_value(x)

        action_dist = distributions.Categorical(logits=pi)

        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)

        return actions, value, log_probs