#!/usr/bin/env python3
import gym, gym_chess, chess
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env
from theseus_v2.wrappers import (
    ChessWrapper,
    SyzygyWrapper,
    AlphaZeroChessWrapper,
    AlphaZeroWrapper2,
    ChessWrapper2,
    TheseusChessWrapper,
)
from theseus_v2.evaluate import Evaluator
from theseus_v2.config import ENV_ID, DEBUG, SYZYGY_ONLY, NO_SYZYGY, NUM_ENVS
from theseus_v2.policy import CustomPolicy
import torch
import numpy as np, random
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)


class Engine:
    """
    Chess 'Theseus' engine class.
    Defines the Theseus engine that is used for training and playing.

    Class attributes:
        path (str): contains the path where the engine is stored.
    """
    path: str = 'ppo_chess'

    def create_model(self, vec_env: DummyVecEnv) -> PPO:
        """
        Create a new PPO model.

        Args:
            vec_env (DummyVecEnv): Vectorized environment.

        Returns:
            PPO: Created PPO model.
        """

        return PPO(CustomPolicy,
            vec_env,
            verbose=1,
            seed=1,
            gamma=0.99,
            n_steps=8192,
            learning_rate=0.0001,
            n_epochs=30,
            tensorboard_log='ppo_tensorboard/',
        )

    def get_model(self, env=None) -> PPO:
        """
        Loads PPO model.

        Returns:
            PPO: Loaded model.
        """
        model = PPO.load(self.path)
        return model

    def make_env(self, env_id: str, evaluator=None) -> Env:
        """
        Creates Gym environment with wrapper.

        Args:
            env_id (str): Gym environment id.
            evaluator (any): Chess position evaluator.

        Returns:
            Env: Gym environment.
        """
        env = gym.make(ENV_ID)
        if env_id == 'syzygy':
            env = SyzygyWrapper(env, None)
        else:
            env = TheseusChessWrapper(env, evaluator)
        return env
    
    def get_callbacks(self,) -> CallbackList:
        ev = Evaluator()
        eval_env = self.make_env(ENV_ID, ev)
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path='./models/',
            name_prefix='ppo_model',
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./logs/',
            log_path='./logs/',
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1,
        )
        callbacks = CallbackList([
            checkpoint_callback,
            eval_callback,
        ])
        return callbacks

    def train(self, total_timesteps=150000) -> None:
        """
        Model training.

        Args:
            total_timesteps (int): Number of timesteps to train the model.
        """
        envs = []
        ev = Evaluator()
        syzygy_env = lambda: self.make_env('syzygy', None)
        training_env = lambda: self.make_env(ENV_ID, ev)
        if SYZYGY_ONLY:
            envs.append(syzygy_env)
        elif NO_SYZYGY:
            for _ in range(NUM_ENVS):
                envs.append(training_env)
        else:
            envs = [training_env, syzygy_env]
        vec_env = DummyVecEnv(envs)
        model = self.create_model(vec_env)
        callback_list = self.get_callbacks()
        model.learn(total_timesteps=total_timesteps,
                    callback=callback_list)

        model.save(self.path)

    def auto_play(self, render=True) -> str:
        """
        Bot-only play.

        Returns:
            str: Exported PGN game.
        """
        env = self.make_env(ENV_ID, evaluator=Evaluator())
        model = self.get_model(env)
        obs = env.reset()
        done = False
        while not done:
            obs = np.array([obs])
            obs = torch.tensor(obs, dtype=torch.float32)
            action = self.predict(model, obs)
            obs, _, done, _ = env.step(action)
            if render: env.render()

        return env.get_pgn()

    def predict(self, model: PPO, obs: torch.Tensor) -> torch.Tensor:
        action, _ = model.policy.predict(obs, None, None, True)
        return action

    def play_against(self, render=True) -> str:
        """
        Play against bot.

        Returns:
            str: Exported PGN game.
        """
        env = self.make_env(ENV_ID, evaluator=Evaluator())
        model = self.get_model(env)
        obs = env.reset()
        done = False
        while not done:
            obs = np.array([obs])
            obs = torch.tensor(obs, dtype=torch.float32)
            action = self.predict(model, obs)
            obs, _, done, _ = env.step(action, bot_only=False)
            if render: env.render()
            retry = True
            b = TheseusChessWrapper.array_to_board(obs)
            while retry:
                moves = list(b.legal_moves)
                move = input('Make a move: ')
                try:
                    move = chess.Move.from_uci(move)
                    if move not in moves:
                        raise Exception
                    obs, _, done, _ = env.env.step(move)
                    obs = env.observation(obs)
                    retry = False
                except:
                    print('Illegal move, try again')
            if render: env.render()

        return env.get_pgn()

if __name__ == '__main__':
    """
    Used mainly for demonstration purposes
    """
    engine = Engine()
    engine.train(total_timesteps=1)
    p = engine.auto_play()
    print(p)
