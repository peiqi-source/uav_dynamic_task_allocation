"""历史版本中的1124groupingPPO 算法脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv

class TargetGroupingEnv(gym.Env):
    """TargetGroupingEnv 类，封装目标grouping环境相关的数据结构与业务行为。

    属性：
        target_groups: 目标groups。
        data_Target: 数据目标。
        action_space: 动作space。
        observation_space: 观测向量space。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            无显式业务参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        super(TargetGroupingEnv, self).__init__()
        # target_groups: 目标groups。
        self.target_groups = {
            1: {'num_targets': 6, 'indices': [0, 1, 8, 18, 20, 36], 'targets': []},
            2: {'num_targets': 3, 'indices': [7, 12, 13], 'targets': []},
            3: {'num_targets': 14, 'indices': [22, 32, 37, 39, 44, 45, 48, 49, 50, 54, 57, 59, 60, 61], 'targets': []},
            4: {'num_targets': 4, 'indices': [3, 11, 19, 21], 'targets': []},
            5: {'num_targets': 4, 'indices': [4, 6, 27, 28], 'targets': []},
            6: {'num_targets': 12, 'indices': [2, 5, 16, 29, 33, 35, 38, 42, 43, 46, 51, 52], 'targets': []},
            7: {'num_targets': 12, 'indices': [9, 10, 14, 23, 34, 40, 41, 47, 53, 55, 56, 58], 'targets': []},
            8: {'num_targets': 7, 'indices': [15, 17, 24, 25, 26, 30, 31], 'targets': []}
        }
        # data_Target: 数据目标。
        self.data_Target = np.loadtxt(open('data_Target_extracted.csv'), delimiter=",", skiprows=1)
        # action_space: 动作space。
        self.action_space = gym.spaces.MultiDiscrete([8, 62, 8])
        num_features = 7
        # observation_space: 观测向量space。
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(8, num_features), dtype=np.float32)

    def _compute_state(self):
        """计算指定指标或中间结果，处理状态相关数据。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        state = []
        num_groups = 8
        num_features = 7
        all_features = np.zeros((num_groups, num_features))
        for group_id in range(1, 9):
            group_data = self.target_groups[group_id]['targets']
            if len(group_data) == 0:
                all_features[group_id - 1] = [0] * 7
                continue
            x_coords = group_data[:, 1]
            y_coords = group_data[:, 2]
            num_targets = len(group_data)
            x_std = np.std(x_coords)
            y_std = np.std(y_coords)
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            distances_to_center = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
            sum_distances = np.sum(distances_to_center)
            importance_sum = np.sum(group_data[:, 5])
            defense_sum = np.sum(group_data[:, 4])
            importance_defense_ratio = importance_sum / defense_sum if defense_sum > 0 else 0
            all_features[group_id - 1] = [num_targets, x_std, y_std, sum_distances, importance_sum, defense_sum,
                                          importance_defense_ratio]

        features_for_z_score = all_features[:, [0, 4, 5, 6]]
        mean_values_z_score = np.mean(features_for_z_score, axis=0)
        std_values_z_score = np.std(features_for_z_score, axis=0)
        for i in range(len(features_for_z_score)):
            for j in range(len(features_for_z_score[i])):
                if std_values_z_score[j] != 0:
                    z_score_value = (features_for_z_score[i][j] - mean_values_z_score[j]) / std_values_z_score[j]
                    features_for_z_score[i][j] = np.tanh(z_score_value)
        features_for_min_max = all_features[:, [1, 2, 3]]
        min_values = np.min(features_for_min_max, axis=0)
        max_values = np.max(features_for_min_max, axis=0)
        for i in range(len(features_for_min_max)):
            for j in range(len(features_for_min_max[i])):
                if max_values[j] - min_values[j] != 0:
                    features_for_min_max[i][j] = (features_for_min_max[i][j] - min_values[j]) / (
                                max_values[j] - min_values[j])
        all_features[:, [0, 4, 5, 6]] = features_for_z_score
        all_features[:, [1, 2, 3]] = features_for_min_max

        all_features = all_features.astype(np.float32)
        # flattened_state = all_features.flatten()

        return all_features

    def reset(self, seed=None, options=None):
        """重置对象状态，为新的回合或流程做准备。

        参数：
            seed: 随机种子。
            options: options 数据。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        super().reset(seed=seed)
        initial_groupings = {
            1: {'num_targets': 6, 'indices': [0, 1, 8, 18, 20, 36], 'targets': []},
            2: {'num_targets': 3, 'indices': [7, 12, 13], 'targets': []},
            3: {'num_targets': 14, 'indices': [22, 32, 37, 39, 44, 45, 48, 49, 50, 54, 57, 59, 60, 61], 'targets': []},
            4: {'num_targets': 4, 'indices': [3, 11, 19, 21], 'targets': []},
            5: {'num_targets': 4, 'indices': [4, 6, 27, 28], 'targets': []},
            6: {'num_targets': 12, 'indices': [2, 5, 16, 29, 33, 35, 38, 42, 43, 46, 51, 52], 'targets': []},
            7: {'num_targets': 12, 'indices': [9, 10, 14, 23, 34, 40, 41, 47, 53, 55, 56, 58], 'targets': []},
            8: {'num_targets': 7, 'indices': [15, 17, 24, 25, 26, 30, 31], 'targets': []}
        }
        for group_id in initial_groupings:
            indices = initial_groupings[group_id]['indices']
            self.target_groups[group_id]['num_targets'] = initial_groupings[group_id]['num_targets']
            self.target_groups[group_id]['indices'] = indices.copy()
            self.target_groups[group_id]['targets'] = self.data_Target[indices]
        return self._compute_state(), {}

    def get_action_mask(self):
        """处理get动作掩码相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        action_mask = np.zeros((8, 62, 8), dtype=bool)

        for source_group_id in range(8):
            source_group_indices = self.target_groups[source_group_id + 1]['indices']
            for target_index in source_group_indices:
                for target_group_id in range(8):
                    if target_group_id!= source_group_id:
                        action_mask[source_group_id, target_index, target_group_id] = True

        return action_mask

    def step(self, action):
        """推进环境或仿真流程的一个时间步。

        参数：
            action: 动作。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        group1, target_index, group2 = action
        if group1 < 1 or group1 > 8 or group2 < 1 or group2 > 8:
            reward = -1
            print("！！！！！！！选择的群组编号不合法！！！！！！！！！！")
            terminated = False
            truncated = False
            return self._compute_state(), reward, terminated, truncated, {}

        source_group = self.target_groups[group1]
        target_group = self.target_groups[group2]

        if target_index < 0 or target_index >= source_group['num_targets']:
            reward = -1
            print("！！！！！！！！选择的目标索引超出源群范围！！！！！！！！")
            terminated = False
            truncated = False
            return self._compute_state(), reward, terminated, truncated, {}

        if group1 == group2:
            reward = -1
            print("！！！！！！！！！选择相同群作为源群和目标群！！！！！！！！！")
            terminated = False
            truncated = False
            return self._compute_state(), reward, terminated, truncated, {}

        if target_group['num_targets'] == 0:
            reward = -1
            print("！！！！！！！目标群为空，无法执行移动操作！！！！！！！")
            terminated = False
            truncated = False
            return self._compute_state(), reward, terminated, truncated, {}

        target_to_move = source_group['targets'][target_index]
        target_index_in_data_Target = source_group['indices'][target_index]
        source_group['indices'] = np.delete(source_group['indices'], target_index)
        source_group['targets'] = np.delete(source_group['targets'], target_index, axis=0)
        source_group['num_targets'] -= 1

        target_group['targets'] = np.vstack([target_group['targets'], target_to_move])
        target_group['num_targets'] += 1
        target_group['indices'] = np.append(target_group['indices'], target_index_in_data_Target)

        reward = self._calculate_reward()

        terminated = False
        truncated = False

        action_mask = self.get_action_mask()

        return self._compute_state(), reward, terminated, truncated, {'action_mask': action_mask}

    def _calculate_reward(self):
        """计算指定指标或中间结果，处理奖励相关数据。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        num_targets_per_group = [group['num_targets'] for group in self.target_groups.values()]
        avg_num_targets = np.mean(num_targets_per_group)
        num_targets_balance_reward = - np.std(num_targets_per_group) / avg_num_targets

        compactness_reward = 0
        for group_id in self.target_groups:
            group_data = self.target_groups[group_id]['targets']
            if len(group_data) > 0:
                x_coords = group_data[:, 1]
                y_coords = group_data[:, 2]
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                distances_to_center = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
                mean_distance_to_center = np.mean(distances_to_center)
                x_std = np.std(x_coords)
                y_std = np.std(y_coords)
                compactness_reward -= 0.5 * (x_std + y_std) - 0.5 * mean_distance_to_center

        importance_defense_ratios = [group['targets'][:, 5].sum() / group['targets'][:, 4].sum() if group['targets'][:, 4].sum() > 0 else 0
                                     for group in self.target_groups.values()]
        avg_ratio = np.mean(importance_defense_ratios)
        task_balance_reward = - np.std(importance_defense_ratios) / avg_ratio if avg_ratio > 0 else 0

        reward = num_targets_balance_reward + compactness_reward + task_balance_reward
        return reward

    def render(self):
        """处理render 数据相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        print("Current target groupings:")
        for group_id in self.target_groups:
            print(f"Group {group_id}: {self.target_groups[group_id]['num_targets']} targets")


class MaskedPPO(PPO):
    """MaskedPPO 类，封装maskedPPO 算法相关的数据结构与业务行为。"""
    def __init__(self, policy, env, **kwargs):
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            policy: policy 参数。
            env: 环境对象。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        super().__init__(policy, env, **kwargs)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        """处理predict 数据相关业务逻辑。

        参数：
            observation: 观测向量。
            state: 状态。
            mask: 掩码。
            deterministic: deterministic 数据。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        if mask is None:
            return super().predict(observation, state, deterministic=deterministic)

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        if state is not None:
            state = torch.as_tensor(state, dtype=torch.float32)

        mask_tensor = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)

        action_distribution, _ = self.policy.forward(obs_tensor, state)

        masked_action_distribution = action_distribution * mask_tensor

        masked_action_distribution = masked_action_distribution / masked_action_distribution.sum(dim=-1, keepdim=True)

        if deterministic:
            _, action = masked_action_distribution.max(dim=-1)
        else:
            action = masked_action_distribution.multinomial(num_samples=1).squeeze(-1)

        return action.detach().numpy(), None


env = TargetGroupingEnv()

# check_env(env)

env = DummyVecEnv([lambda: env])

# 创建PPO模型
model = PPO("MlpPolicy", env, gamma=0.95, learning_rate=1e-4, ent_coef=0.01, verbose=1)
# 训练
model.learn(total_timesteps=10000)

obs = env.reset()[0]
for _ in range(100):
    action, _states = model.predict(obs)
    print("Model predicted action:", action)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
