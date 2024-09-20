#!/usr/bin/env python3
import torch, chess
from stable_baselines3.ppo import MlpPolicy
from theseus_v2.wrappers import TheseusChessWrapper, AlphaZeroWrapper2
from theseus_v2.board import encode_move


class CustomPolicy(MlpPolicy):
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple:
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        values = self.value_net(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        action_mask = self.get_action_mask(obs)
        action_probs = distribution.distribution.logits
        masked_logits = action_probs * (action_mask + 1e-45).log()

        masked_probs = torch.softmax(masked_logits, dim=1)
        if deterministic:
            actions = masked_probs.argmax(dim=1)
        else:
            actions = masked_probs.multinomial(num_samples=1).squeeze()
        
        log_prob = torch.log(masked_probs.gather(1, actions.unsqueeze(1))).squeeze()

        return actions, values, log_prob


    def get_action_mask(self, obs):
        batch_size = len(obs)
        masks = torch.ones((batch_size, self.action_space.n), dtype=torch.float32)

        for i in range(batch_size):
            _, moves = TheseusChessWrapper.array_to_board(obs[i])
            for move in moves:
                masks[i, encode_move(chess.Move.from_uci(move))] = 0
        return masks

    def predict(self, obs, a, b, deterministic=False):
        actions, _, _ = self.forward(obs, deterministic)
        return actions, _