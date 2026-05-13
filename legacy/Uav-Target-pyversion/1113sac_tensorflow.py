"""历史版本中的1113sactensorflow脚本，保留用于算法对照、复现实验或迁移参考。"""
import gymnasium
import numpy as np
from gymnasium import spaces
import math
from stable_baselines3 import SAC

class TargetGroupingEnv(gymnasium.Env):
    """TargetGroupingEnv 类，封装目标grouping环境相关的数据结构与业务行为。

    属性：
        data_Target: 数据目标。
        num_targets: num目标集合。
        num_groups: numgroups。
        action_space: 动作space。
        observation_space: 观测向量space。
        current_grouping: 当前grouping。
        target_counts: 目标counts。
    """
    def __init__(self, data_Target):
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            data_Target: data_Target 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        super(TargetGroupingEnv, self).__init__()
        # data_Target: 数据目标。
        self.data_Target = data_Target[:, 1:3].astype(float)
        # num_targets: num目标集合。
        self.num_targets = len(self.data_Target)
        # num_groups: numgroups。
        self.num_groups = 8
        # action_space: 动作space。
        self.action_space = spaces.Discrete(self.num_targets * self.num_groups)
        # observation_space: 观测向量space。
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_targets, 2), dtype=np.float32)

        # current_grouping: 当前grouping。
        self.current_grouping = np.zeros(self.num_targets, dtype=int)
        # target_counts: 目标counts。
        self.target_counts = np.zeros(self.num_groups, dtype=int)

    def reset(self):
        """重置对象状态，为新的回合或流程做准备。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        self.current_grouping = np.random.choice(self.num_groups, self.num_targets)
        self.target_counts = np.bincount(self.current_grouping, minlength=self.num_groups)
        return self.data_Target.copy()

    def step(self, action):
        """推进环境或仿真流程的一个时间步。

        参数：
            action: 动作。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
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
        """计算指定指标或中间结果，处理compactness奖励相关数据。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
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
        """计算指定指标或中间结果，处理distance 数据相关数据。

        参数：
            target1: 目标1。
            target2: 目标2。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
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