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

        masked_logits = distribution.distribution.logits.clone()
        masked_logits[action_mask == 0] = -float('inf')

        masked_probs = torch.softmax(masked_logits, dim=1)
        #masked_probs /= masked_probs.sum(dim=1, keepdim=True)
        if deterministic:
            actions = masked_probs.argmax(dim=1)
        else:
            try:
                actions = masked_probs.multinomial(num_samples=1).squeeze()
            except:
                actions = masked_probs.argmax(dim=1)

        log_prob = torch.log(masked_probs.gather(1, actions.unsqueeze(1))).squeeze()

        return actions, values, log_prob

    def get_action_mask(self, obs):
        batch_size = len(obs)
        masks = torch.zeros((batch_size, self.action_space.n), dtype=torch.float32)

        for i in range(batch_size):
            b = TheseusChessWrapper.array_to_board(obs[i])
            for move in b.legal_moves:
                masks[i, encode_move(move)] = 1
        return masks

    def predict(self, obs, deterministic=False):
        actions, _, _ = self.forward(obs, deterministic)
        return actions, _
