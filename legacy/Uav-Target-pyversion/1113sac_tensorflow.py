import gymnasium
import numpy as np
from gymnasium import spaces
import math
from stable_baselines3 import SAC

class TargetGroupingEnv(gymnasium.Env):
    def __init__(self, data_Target):
        super(TargetGroupingEnv, self).__init__()
        self.data_Target = data_Target[:, 1:3].astype(float)
        self.num_targets = len(self.data_Target)
        self.num_groups = 8
        self.action_space = spaces.Discrete(self.num_targets * self.num_groups)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_targets, 2), dtype=np.float32)

        self.current_grouping = np.zeros(self.num_targets, dtype=int)
        self.target_counts = np.zeros(self.num_groups, dtype=int)

    def reset(self):
        self.current_grouping = np.random.choice(self.num_groups, self.num_targets)
        self.target_counts = np.bincount(self.current_grouping, minlength=self.num_groups)
        return self.data_Target.copy()

    def step(self, action):
        target_index = action // self.num_groups
        group_index = action % self.num_groups

        prev_group = self.current_grouping[target_index]
        self.current_grouping[target_index] = group_index
        self.target_counts[prev_group] -= 1
        self.target_counts[group_index] += 1

        avg_count = self.num_targets / self.num_groups
        count_reward = - np.sum(np.abs(self.target_counts - avg_count))
        distance_reward = self.calculate_compactness_reward()
        reward = count_reward + distance_reward

        done = False
        info = {}
        return self.data_Target.copy(), reward, done, info

    def calculate_compactness_reward(self):
        compactness_reward = 0
        for group in range(self.num_groups):
            group_indices = np.where(self.current_grouping == group)[0]
            if len(group_indices) > 0:
                group_targets = self.data_Target[group_indices]
                distances = []
                for i in range(len(group_targets)):
                    for j in range(i + 1, len(group_targets)):
                        dist = self.calculate_distance(group_targets[i], group_targets[j])
                        distances.append(dist)
                avg_distance = np.mean(distances) if distances else 0
                compactness_reward -= avg_distance  # 距离越小奖励越高
        return compactness_reward

    def calculate_distance(self, target1, target2):
        return math.sqrt((target1[0] - target2[0]) ** 2 + (target2[1] - target1[1]) ** 2)


if __name__ == "__main__":
    data_Target = np.loadtxt(open('data_Target_extracted.csv'), delimiter=",", skiprows=1)

    env = TargetGroupingEnv(data_Target)

    model = SAC("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=10000)

    obs = env.reset()
    for _ in range(len(data_Target)):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
    optimal_grouping = env.current_grouping

    for group in range(env.num_groups):
        group_indices = np.where(optimal_grouping == group)[0]
        print(f"Group {group} targets: {group_indices}")