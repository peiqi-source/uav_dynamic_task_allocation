import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv


class ClusteringEnv(gym.Env):
    def __init__(self, target_data, num_clusters=3):
        super(ClusteringEnv, self).__init__()
        self.target_data = target_data
        self.num_clusters = num_clusters
        self.num_targets = len(target_data)

        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_targets, 6), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=self.num_clusters - 1, shape=(1,), dtype=np.float32)  

        
        self.clusters = [[] for _ in range(self.num_clusters)]
        self.cluster_centers = [np.zeros(2) for _ in range(self.num_clusters)]

    def reset(self, seed=None, **kwargs):
        self.clusters = [[] for _ in range(self.num_clusters)]
        self.cluster_centers = [np.zeros(2) for _ in range(self.num_clusters)]
        if seed is not None:
            np.random.seed(seed)  
        return np.array(self.target_data), {}  

    def step(self, action):
        
        action_idx = int(np.clip(action[0], 0, self.num_clusters - 1))
        target_idx = np.random.choice(range(self.num_targets))
        target = self.target_data[target_idx]
        self.clusters[action_idx].append(target)
        self.update_cluster_center(action_idx)

        reward = self.calculate_compactness_reward(action_idx) + self.calculate_balance_reward()

        
        print(f"Step reward: {reward}, action: {action_idx}")

        done = len(sum(self.clusters, [])) == self.num_targets  

        return np.array(self.target_data), reward, done, False, {}  

    def update_cluster_center(self, cluster_idx):
        points = np.array(self.clusters[cluster_idx])
        if len(points) > 0:
            self.cluster_centers[cluster_idx] = np.mean(points[:, 1:3], axis=0)

    def calculate_compactness_reward(self, cluster_idx):
        points = np.array(self.clusters[cluster_idx])
        if len(points) > 1:
            distances = np.linalg.norm(points[:, 1:3] - self.cluster_centers[cluster_idx], axis=1)
            compactness_reward = -np.mean(distances)
        else:
            compactness_reward = 0
        return compactness_reward

    def calculate_balance_reward(self):
        sizes = [len(cluster) for cluster in self.clusters]
        if max(sizes) - min(sizes) > 1:
            balance_reward = -np.std(sizes)
        else:
            balance_reward = 0
        return balance_reward


target_data = np.array([
    [1, -388, 1090, 1, 3, 2],
    [2, -479, 999, 1, 4, 2],
    [3, 156, 6, 1, 3, 2],
    [4, 724, 665, 2, 3, 2],
    [5, 982, -594, 1, 4, 2],
    [6, 885, -386, 1, 3, 2],
    [7, 1138, -1322, 2, 1, 2],
    [8, -1871, 1048, 1, 2, 2],
    [9, -943, 735, 2, 4, 2],
    [10, -405, -1017, 2, 2, 2]
])

env = DummyVecEnv([lambda: ClusteringEnv(target_data)])

model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(len(target_data)):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        break


for idx, cluster in enumerate(env.get_attr('clusters')[0]):
    print(f"Cluster {idx + 1}: {cluster}")