import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TargetClusterEnv(gym.Env):
    def __init__(self, data_Target, num_clusters=8):
        super(TargetClusterEnv, self).__init__()
        self.data = data_Target
        self.num_clusters = num_clusters
        self.num_targets = len(data_Target)
        self.state = self._initialize_clusters()
        self.action_space = spaces.MultiDiscrete([num_clusters, self.num_targets, num_clusters])
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(num_clusters * 6,), dtype=np.float32)

    def _initialize_clusters(self):
        return [list() for _ in range(self.num_clusters)]

    def step(self, action):
        G1, T, G2 = action
        if T not in self.state[G1] or G1 == G2:
            return self.state, -10, False, {}
        self.state[G1].remove(T)
        self.state[G2].append(T)
        reward = self._compute_reward()
        return self._compute_state_features(), reward, False, {}

    def reset(self):
        self.state = self._initialize_clusters()
        return self._compute_state_features()

    def _compute_state_features(self):
        features = []
        for group in self.state:
            indices = np.array(group)
            features.extend(self._compute_group_features(indices))
        return np.array(features)

    def _compute_group_features(self, indices):
        if len(indices) == 0:
            return [0] * 6
        coords = self.data[indices, 1:3]
        importance = self.data[indices, -1]
        defense = self.data[indices, -2]
        return [
            len(indices),
            np.std(coords[:, 0]),
            np.std(coords[:, 1]),
            np.mean(np.linalg.norm(coords - np.mean(coords, axis=0), axis=1)),
            np.sum(importance),
            np.sum(defense)
        ]

    def _compute_reward(self):
        group_sizes = [len(g) for g in self.state]
        importance_defense_ratios = [
            np.sum(self.data[g, -1]) / max(np.sum(self.data[g, -2]), 1) for g in self.state if len(g) > 0
        ]
        return (
            -np.std(group_sizes)
            -np.std([np.std(self.data[g, 1]) for g in self.state if len(g) > 0])
            -np.std([np.std(self.data[g, 2]) for g in self.state if len(g) > 0])
            -np.std(importance_defense_ratios)  # 任务均衡
        )

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

data_Target = np.loadtxt(open('data_Target_extracted.csv'), delimiter=",", skiprows=1)
env = make_vec_env(lambda: TargetClusterEnv(data_Target), n_envs=1)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100000)

model.save("target_cluster_ppo")

model = PPO.load("target_cluster_ppo")

obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}")
