#!/usr/bin/env python3
import gym, gym_chess
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env
from wrappers import ChessWrapper


ENV_ID = 'Chess-v0'
NUM_ENVS = 1


class Engine:
    path = 'ppo_chess'

    def create_model(self, vec_env: DummyVecEnv) -> PPO:
        """
        Create a new PPO model.

        Args:
            vec_env (DummyVecEnv): Vectorized environment.

        Returns:
            PPO: Created PPO model.
        """
        return PPO('MlpPolicy', vec_env, verbose=1)

    def get_model(self) -> PPO:
        """
        Loads PPO model.

        Returns:
            PPO: Loaded model.
        """
        model = PPO.load(self.path)
        return model

    def make_env(self, env_id: str) -> Env:
        """
        Creates Gym environment with wrapper.

        Args:
            env_id (str): Gym environment id.

        Returns:
            Env: Gym environment.
        """
        env = gym.make(env_id)
        env = ChessWrapper(env)
        return env

    def train(self, total_timesteps=150000) -> None:
        """
        Model training.

        Args:
            total_timesteps (int): Number of timesteps to train the model.
        """
        envs = [lambda: self.make_env(ENV_ID) for _ in range(NUM_ENVS)]
        vec_env = DummyVecEnv(envs)
        model = self.create_model(vec_env)
        model.learn(total_timesteps=total_timesteps)

        model.save(self.path)

    def auto_play(self) -> int:
        """
        Bot-only play.

        Returns:
            int: Total rewards.
        """
        model = self.get_model()
        env = self.make_env(ENV_ID)
        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()

        return episode_reward

if __name__ == '__main__':
    engine = Engine()
    engine.train(total_timesteps=1)
    engine.auto_play()
