"""历史版本中的sacgrouping1脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv


class ClusteringEnv(gym.Env):
    """ClusteringEnv 类，封装clustering环境相关的数据结构与业务行为。

    属性：
        target_data: 目标数据。
        num_clusters: num目标簇集合。
        num_targets: num目标集合。
        observation_space: 观测向量space。
        action_space: 动作space。
        clusters: 目标簇集合。
        cluster_centers: 目标簇centers。
    """
    def __init__(self, target_data, num_clusters=3):
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            target_data: target_data 参数。
            num_clusters: num_clusters 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        super(ClusteringEnv, self).__init__()
        # target_data: 目标数据。
        self.target_data = target_data
        # num_clusters: num目标簇集合。
        self.num_clusters = num_clusters
        # num_targets: num目标集合。
        self.num_targets = len(target_data)


        # observation_space: 观测向量space。
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_targets, 6), dtype=np.float32)
        # action_space: 动作space。
        self.action_space = spaces.Box(low=0, high=self.num_clusters - 1, shape=(1,), dtype=np.float32)


        # clusters: 目标簇集合。
        self.clusters = [[] for _ in range(self.num_clusters)]
        # cluster_centers: 目标簇centers。
        self.cluster_centers = [np.zeros(2) for _ in range(self.num_clusters)]

    def reset(self, seed=None, **kwargs):
        """重置对象状态，为新的回合或流程做准备。

        参数：
            seed: 随机种子。
            **kwargs: 可变关键字参数，用于透传扩展配置。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        self.clusters = [[] for _ in range(self.num_clusters)]
        self.cluster_centers = [np.zeros(2) for _ in range(self.num_clusters)]
        if seed is not None:
            np.random.seed(seed)
        return np.array(self.target_data), {}

    def step(self, action):

        """推进环境或仿真流程的一个时间步。

        参数：
            action: 动作。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
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
        """处理update目标簇center相关业务逻辑。

        参数：
            cluster_idx: 目标簇idx。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        points = np.array(self.clusters[cluster_idx])
        if len(points) > 0:
            self.cluster_centers[cluster_idx] = np.mean(points[:, 1:3], axis=0)

    def calculate_compactness_reward(self, cluster_idx):
        """计算指定指标或中间结果，处理compactness奖励相关数据。

        参数：
            cluster_idx: 目标簇idx。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        points = np.array(self.clusters[cluster_idx])
        if len(points) > 1:
            distances = np.linalg.norm(points[:, 1:3] - self.cluster_centers[cluster_idx], axis=1)
            compactness_reward = -np.mean(distances)
        else:
            compactness_reward = 0
        return compactness_reward

    def calculate_balance_reward(self):
        """计算指定指标或中间结果，处理balance奖励相关数据。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
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