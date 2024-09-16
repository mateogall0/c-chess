#!/usr/bin/env python3
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


class CustomMlpPolicy(ActorCriticPolicy):
    def forward(self, obs, deterministic=False):
        latent_pi, latent_vf = self._get_latent(obs)

        distribution = self._get_action_dist_from_latent(latent_pi)

        action = distribution.get_deterministic_action() if deterministic else distribution.sample()

        return action
