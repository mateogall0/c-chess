#!/usr/bin/env python3
import torch
from stable_baselines3.ppo import MlpPolicy
from theseus_v2.wrappers import ChessWrapper2

class CustomPolicy(MlpPolicy):
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple:
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        action_mask = self.get_action_mask(obs)
        action_probs = distribution.distribution.logits
        masked_probs = action_probs * action_mask
        masked_probs = masked_probs / masked_probs.sum(dim=1, keepdim=True)
        actions = masked_probs.argmax(dim=1) if deterministic else masked_probs.multinomial(num_samples=1).squeeze()
        log_prob = torch.log(masked_probs.gather(1, actions.unsqueeze(1))).squeeze()
        
        return actions, values, log_prob

    def get_action_mask(self, obs):
        batch_size = len(obs)
        masks = torch.zeros((batch_size, self.action_space.n), dtype=torch.float32)
        for i in range(batch_size):
            _, valid_moves = ChessWrapper2.array_to_board(obs[i])
            num_valid_moves = len(valid_moves)
            masks[i, :num_valid_moves] = 1
        return masks
