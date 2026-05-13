"""历史版本中的wuyonghuatu脚本，保留用于算法对照、复现实验或迁移参考。"""
import gymnasium as gym
from stable_baselines3 import A2C
import matplotlib.pyplot as plt
import numpy as np


env = gym.make("CartPole-v1", render_mode="rgb_array")


model = A2C("MlpPolicy", env, verbose=1)


total_timesteps = 10000


all_rewards = []


model.learn(total_timesteps=total_timesteps)


obs, _ = env.reset()
episode_reward = 0
while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    episode_reward += reward
    if terminated or truncated:
        all_rewards.append(episode_reward)
        episode_reward = 0
        obs, _ = env.reset()
    if len(all_rewards) >= 1000:
        break


plt.plot(np.arange(len(all_rewards)), all_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("CartPole Reward Curve")
plt.show()


env.close()
