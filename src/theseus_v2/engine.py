#!/usr/bin/env python3
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = 'CartPole-v1'
NUM_ENVS = 1

class Engine:
    path = 'ppo_chess'

    def make_env(self, env_id):
        return gym.make(env_id)

    def train(self):
        envs = [lambda: self.make_env(ENV_ID) for _ in range(NUM_ENVS)]
        vec_env = DummyVecEnv(envs)
        model = PPO('MlpPolicy', vec_env, verbose=1)
        model.learn(total_timesteps=20000)

        model.save(self.path)

    def play(self):
        model = PPO.load(self.path)
        env = self.make_env(ENV_ID)
        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()

        print(f"Total reward: {episode_reward}")

if __name__ == '__main__':
    engine = Engine()
    engine.train()
    engine.play()