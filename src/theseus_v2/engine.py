#!/usr/bin/env python3
import gym, gym_chess
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = 'Chess-v0'
NUM_ENVS = 1

from theseus_v2.wrappers import ChessWrapper

class Engine:
    path = 'ppo_chess'

    def get_model(self):
        model = PPO.load(self.path)
        return model

    def make_env(self, env_id):
        env = gym.make(env_id)
        env = ChessWrapper(env)
        return env

    def train(self, total_timesteps=150000):
        envs = [lambda: self.make_env(ENV_ID) for _ in range(NUM_ENVS)]
        vec_env = DummyVecEnv(envs)
        model = PPO('MlpPolicy', vec_env, verbose=1)
        model.learn(total_timesteps=total_timesteps)

        model.save(self.path)

    def play(self):
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

if __name__ == '__main__':
    engine = Engine()
    engine.train()
    engine.play()