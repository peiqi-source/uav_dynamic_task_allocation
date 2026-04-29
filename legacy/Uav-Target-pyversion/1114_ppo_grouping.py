import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium

NUM_TARGETS = 62

NUM_GROUPS = 8
AVG_TARGETS_PER_GROUP = NUM_TARGETS // NUM_GROUPS

def std_x(target_indices, data_Target):
    if len(target_indices) == 0:
        return 0
    return np.std(data_Target[target_indices, 1])

def std_y(target_indices, data_Target):
    if len(target_indices) == 0:
        return 0
    return np.std(data_Target[target_indices, 2])

def distance_to_center_sum(target_indices, data_Target):
    if len(target_indices) == 0:
        return 0
    x_mean = np.mean(data_Target[target_indices, 1])
    y_mean = np.mean(data_Target[target_indices, 2])
    distances = np.sqrt((data_Target[target_indices, 1] - x_mean) ** 2 + (data_Target[target_indices, 2] - y_mean) ** 2)
    return np.sum(distances)


class TargetGroupingEnv(gymnasium.Env):
    def __init__(self, data_Target):
        super(TargetGroupingEnv, self).__init__()
        self.data_Target = data_Target
        self.current_grouping = self.initialize_grouping()
        self.action_space = gymnasium.spaces.Discrete(NUM_GROUPS * NUM_GROUPS)  # 所有可能的转移动作
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(self.get_obs_size(),), dtype=np.float32)

    def initialize_grouping(self):
        current_grouping = [[] for _ in range(NUM_GROUPS)]
        groups = [
            [0, 1, 8, 18, 20, 36],
            [7, 12, 13],
            [22, 32, 37, 39, 44, 45, 48, 49, 50, 54, 57, 59, 60, 61],
            [3, 11, 19, 21],
            [4, 6, 27, 28],
            [2, 5, 16, 29, 33, 35, 38, 42, 43, 46, 51, 52],
            [9, 10, 14, 23, 34, 40, 41, 47, 53, 55, 56, 58],
            [15, 17, 24, 25, 26, 30, 31]
        ]
        for i, group_indices in enumerate(groups):
            assert group_indices, f"Group {i} is initialized as empty!"
            current_grouping[i] = group_indices
        return current_grouping

    def get_obs_size(self):
        return NUM_GROUPS * 4

    def get_obs(self):
        obs = []
        for group in self.current_grouping:
            num_diff = len(group) - AVG_TARGETS_PER_GROUP
            std_x_value = std_x(group, self.data_Target)
            std_y_value = std_y(group, self.data_Target)
            dist_center_sum = distance_to_center_sum(group, self.data_Target)

            obs.append([num_diff, std_x_value, std_y_value, dist_center_sum])

        obs = np.array(obs)
        print("Before normalization:", obs)
        for col in range(obs.shape[1]):
            min_value = np.min(obs[:, col])
            max_value = np.max(obs[:, col])
            if max_value == min_value:
                obs[:, col] = 0
            else:
                obs[:, col] = (obs[:, col] - min_value) / (max_value - min_value)

        print("After normalization:", obs)
        return obs.flatten()

    def step(self, action):
        source_group = action // NUM_GROUPS
        target_group = action % NUM_GROUPS
        target_to_move = self.current_grouping[source_group][0]
        self.current_grouping[source_group].remove(target_to_move)
        self.current_grouping[target_group].append(target_to_move)

        reward = self.calculate_reward()
        done = False
        truncated = False
        obs = self.get_obs()
        return obs, reward, done, truncated, {}

    def calculate_reward(self):
        num_diff_reward = 0
        compactness_reward = 0
        for group in self.current_grouping:
            num_diff = len(group) - AVG_TARGETS_PER_GROUP
            num_diff_reward -= abs(num_diff)
            std_x_value = std_x(group, self.data_Target)
            std_y_value = std_y(group, self.data_Target)
            compactness_reward -= (std_x_value + std_y_value)
        return (num_diff_reward + compactness_reward) / (2 * NUM_GROUPS)

    def reset(self, seed=None):
        self.current_grouping = self.initialize_grouping()
        obs = self.get_obs()
        return obs, {}


data_Target = np.loadtxt(open('data_Target_extracted.csv'), delimiter=",", skiprows=1)

env = TargetGroupingEnv(data_Target)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

optimized_grouping = env.current_grouping
print(optimized_grouping)

