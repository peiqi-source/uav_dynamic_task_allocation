"""历史版本中的1125groupingPPO 算法脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TargetClusterEnv(gym.Env):
    """TargetClusterEnv 类，封装目标目标簇环境相关的数据结构与业务行为。

    属性：
        data: 数据。
        num_clusters: num目标簇集合。
        num_targets: num目标集合。
        state: 状态。
        action_space: 动作space。
        observation_space: 观测向量space。
    """
    def __init__(self, data_Target, num_clusters=8):
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            data_Target: data_Target 参数。
            num_clusters: num_clusters 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        super(TargetClusterEnv, self).__init__()
        # data: 数据。
        self.data = data_Target
        # num_clusters: num目标簇集合。
        self.num_clusters = num_clusters
        # num_targets: num目标集合。
        self.num_targets = len(data_Target)
        # state: 状态。
        self.state = self._initialize_clusters()
        # action_space: 动作space。
        self.action_space = spaces.MultiDiscrete([num_clusters, self.num_targets, num_clusters])
        # observation_space: 观测向量space。
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(num_clusters * 6,), dtype=np.float32)

    def _initialize_clusters(self):
        """处理initialize目标簇集合相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        return [list() for _ in range(self.num_clusters)]

    def step(self, action):
        """推进环境或仿真流程的一个时间步。

        参数：
            action: 动作。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        G1, T, G2 = action
        if T not in self.state[G1] or G1 == G2:
            return self.state, -10, False, {}
        self.state[G1].remove(T)
        self.state[G2].append(T)
        reward = self._compute_reward()
        return self._compute_state_features(), reward, False, {}

    def reset(self):
        """重置对象状态，为新的回合或流程做准备。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        self.state = self._initialize_clusters()
        return self._compute_state_features()

    def _compute_state_features(self):
        """计算指定指标或中间结果，处理状态features相关数据。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        features = []
        for group in self.state:
            indices = np.array(group)
            features.extend(self._compute_group_features(indices))
        return np.array(features)

    def _compute_group_features(self, indices):
        """计算指定指标或中间结果，处理groupfeatures相关数据。

        参数：
            indices: indices 数据。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
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
        """计算指定指标或中间结果，处理奖励相关数据。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
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
