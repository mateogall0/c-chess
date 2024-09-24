#!/usr/bin/env python3
import torch, chess
from stable_baselines3.ppo import MlpPolicy
from theseus_v2.wrappers import TheseusChessWrapper, AlphaZeroWrapper2
from theseus_v2.board import encode_move


class CustomPolicy(MlpPolicy):
    """
    Custom policy used for Theseus Chess bot.
    Inherits from the MlpPolicy from Stable Baselines 3.
    This policy incorporates a mask used to handle illegal Chess moves. Uses
    logits to calculate action probabilities while applying the action mask.
    """
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple:
        """
        Forward pass of the policy network.

        Using `_get_latent`, the latent representation of both policy
        (latent_pi) and value (latent_vf), used to predict actions and used to
        estimate the value of the current state.

        The value prediction is a value given by `value_net` using `latent_vf`
        to estimate the value of the state.

        Using `latent_pi` the method creates a distribution over possible
        actions.

        :Actions masking: A mask is applied to filter out illegal actions in
        current state, ensuring that the model only considers legal actions.
        The action scores (logits) are ajusted so that illegal moves have
        extremely low probabilites (-inf).
        
        After masking the logits are passed through softmax to create a valid
        probability distribution for legal moves. This ensures the model only
        selects legal actions.

        :Action selection: If deterministic, the model chooses the action with
        highest probability. If stochastic, it samples an action based on the
        computed probabilities.

        Returns:
            Computed log probability of the selected action as:
                (action, value estimate, log probability).
        """
        # pass the observation through neural network layers
        # generates intermediate activations (latents)
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)

        # compute value predictions
        values = self.value_net(latent_vf)

        # get the action distribution by  using the latent policy representation
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)

        action_mask = self.get_action_mask(obs)

        # clone logits
        # avoids in-place modification
        masked_logits = distribution.distribution.logits.clone()

        # set logits of illegal moves to -inf
        # excludes them from being selected
        masked_logits[action_mask == 0] = -float('inf')

        # clamp logits to prevent large or small values
        masked_logits = torch.clamp(masked_logits, min=-1e10, max=1e10)

        # compute probabilities using softmax to the masked logits
        masked_probs = torch.softmax(masked_logits, dim=1)

        # handle cases where may be nan values
        masked_probs[torch.isnan(masked_probs)] = 0.0

        # handle cases where the sum of probabilities for normalization
        # each row should sum to 1
        masked_probs_sum = masked_probs.sum(dim=1, keepdim=True)
        zero_sum_mask = masked_probs_sum == 0.0
        if zero_sum_mask.any():
            print("Warning: zero probability sum detected, handling fallback")
            masked_probs[zero_sum_mask] = action_mask[zero_sum_mask] / action_mask[zero_sum_mask].sum(dim=1, keepdim=True)

        # normalize the masked probabilities
        # ensures they sum to 1 for each observation
        masked_probs /= masked_probs.sum(dim=1, keepdim=True)

        if deterministic:
            # greedy choice for highest probability
            actions = masked_probs.argmax(dim=1)
        else:
            try:
                actions = masked_probs.multinomial(num_samples=1).squeeze()
            except:
                # in case of an error during sampling fall back to
                # selecting the action with the highest probability
                # deterministically
                actions = masked_probs.argmax(dim=1)

        # for PPO objective, compute the log probability of the
        # selected actions
        log_prob = torch.log(masked_probs.gather(1, actions.unsqueeze(1)).clamp(min=1e-10)).squeeze()

        return actions, values, log_prob

    def get_action_mask(self, obs: torch.ensor) -> torch.Tensor:
        """
        Action mask creator.
        Encodes the moves using an AlphaZero encoder-like that signs legal
        actions as ones.
        The legal actions are obtained from the board of the given position
        looping every batch. The board is retrieved from the observation.
        """
        batch_size = len(obs)
        masks = torch.zeros((batch_size, self.action_space.n), dtype=torch.float32)

        for i in range(batch_size):
            # turn obs back to a board
            b = TheseusChessWrapper.array_to_board(obs[i])
            for move in b.legal_moves:
                masks[i, encode_move(move)] = 1
        return masks

    def predict(self, obs: torch.Tensor, deterministic=False):
        """
        Predict action for given observation.
        """
        actions, _, _ = self.forward(obs, deterministic)
        return actions
