"""历史版本中的无人机毁伤状态脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import random
import gym
from gym import spaces
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DroneEnv:
    """DroneEnv 类，封装无人机环境相关的数据结构与业务行为。

    属性：
        targets: 目标集合。
        drones: drones 数据。
        num_clusters: num目标簇集合。
        clusters: 目标簇集合。
        num_targets: num目标集合。
        num_drones: numdrones。
        state: 状态。
        observation_space: 观测向量space。
        action_space: 动作space。
    """
    def __init__(self, targets, drones, clusters):
        # 说明：历史脚本沿用早期变量命名，含义请结合上下文和算法流程理解。
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            targets: targets 参数。
            drones: drones 参数。
            clusters: clusters 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        # targets: 目标集合。
        self.targets = targets
        # drones: drones 数据。
        self.drones = drones
        # num_clusters: num目标簇集合。
        self.num_clusters = len(np.unique(clusters))
        # clusters: 目标簇集合。
        self.clusters = clusters
        # num_targets: num目标集合。
        self.num_targets = len(targets)
        # num_drones: numdrones。
        self.num_drones = len(drones)
        # state: 状态。
        self.state = self.reset()
        # observation_space: 观测向量space。
        self.observation_space = self.state.shape
        # action_space: 动作space。
        self.action_space = np.array([self.num_drones] * self.num_targets)

    def reset(self):
        """重置对象状态，为新的回合或流程做准备。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        self.state = np.zeros((self.num_targets, 2))
        return self.state

    def step(self, action):
        """推进环境或仿真流程的一个时间步。

        参数：
            action: 动作。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        if not self._validate_action(action):
            reward = -float('inf')
            done = True
            return self.state, reward, done, {}

        reward = self._compute_reward(action)
        done = self._is_done()
        self.state = self._update_state(action)
        return self.state, reward, done, {}

    def _validate_action(self, action):
        """校验当前对象或输入配置的合法性，处理动作相关数据。

        参数：
            action: 动作。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        cluster_assignments = [[] for _ in range(self.num_clusters)]
        for target_id, drone_id in enumerate(action):
            cluster_id = self.clusters[target_id]
            cluster_assignments[cluster_id].append(drone_id)

        for assignments in cluster_assignments:
            if len(assignments) == 0 or not any(self.drones[drone_id] == 'recon' for drone_id in assignments):
                return False
        return True

    def _compute_reward(self, action):
        """计算指定指标或中间结果，处理奖励相关数据。

        参数：
            action: 动作。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        clusters = [[] for _ in range(self.num_clusters)]
        for idx, cluster_id in enumerate(action):
            clusters[cluster_id].append(self.targets[idx])

        total_value = sum(sum(target[2] for target in cluster) for cluster in clusters)

        cluster_sizes = [len(cluster) for cluster in clusters]
        max_size = max(cluster_sizes)
        min_size = min(cluster_sizes)
        balance_penalty = max_size - min_size

        angle_penalty = 0
        for cluster in clusters:
            if len(cluster) > 1:
                angles = [target[5] for target in cluster]
                mean_angle = np.mean(angles)
                angle_penalty += sum(abs(angle - mean_angle) for angle in angles)

        return -total_value + balance_penalty + angle_penalty

    def _is_done(self):

        """处理is结束标记相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        return False

    def _update_state(self, action):

        """处理update状态相关业务逻辑。

        参数：
            action: 动作。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        self.state = np.zeros((self.num_targets, 2))
        for i in range(self.num_targets):
            self.state[i, 0] = action[i]
            self.state[i, 1] = self.targets[i, 2]
        return self.state

class DQNAgent:
    """DQNAgent 类，封装DQN 算法智能体相关的数据结构与业务行为。

    属性：
        state_size: 状态size。
        action_size: 动作size。
        memory: memory 数据。
        gamma: 折扣因子。
        epsilon: 探索率。
        epsilon_min: 探索率最小值。
        epsilon_decay: 探索率decay。
        learning_rate: 学习率。
        model: 模型。
    """
    def __init__(self, state_size, action_size):
        # state_size: 状态向量维度。
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            state_size: state_size 参数。
            action_size: action_size 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        # state_size: 状态size。
        self.state_size = state_size
        # action_size: 动作size。
        self.action_size = action_size
        # memory: memory 数据。
        self.memory = deque(maxlen=2000)
        # gamma: 折扣因子。
        self.gamma = 0.95
        # epsilon: 探索率。
        self.epsilon = 1.0
        # epsilon_min: 探索率最小值。
        self.epsilon_min = 0.01
        # epsilon_decay: 探索率decay。
        self.epsilon_decay = 0.995
        # learning_rate: 学习率。
        self.learning_rate = 0.001
        # model: 模型。
        self.model = self._build_model()

    def _build_model(self):
        """构建后续流程需要的领域对象或配置对象，处理模型相关数据。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """处理remember 数据相关业务逻辑。

        参数：
            state: 状态。
            action: 动作。
            reward: 奖励。
            next_state: 下一步状态。
            done: 结束标记。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """处理act 数据相关业务逻辑。

        参数：
            state: 状态。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        if np.random.rand() <= self.epsilon:
            return random.sample(range(self.action_size), self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """处理replay 数据相关业务逻辑。

        参数：
            batch_size: 批量样本size。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """从配置文件或外部数据源加载所需数据。

        参数：
            name: 名称。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        self.model.load_weights(name)

    def save(self, name):
        """将对象、模型或指标保存到指定位置。

        参数：
            name: 名称。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        self.model.save_weights(name)


def main():
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    targets = np.array([
        [1, 0, 10, 1, 2, 45],
        [2, 0, 20, 1, 2, 30],
        [3, 0, 30, 1, 2, 60],

    ])
    drones = np.array(['recon', 'attack', 'recon'])

    clusters = np.array([0, 1, 2])

    env = DroneEnv(targets, drones, clusters)
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.shape[0]
    agent = DQNAgent(state_size, action_size)

    episodes = 1000
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}".format(e, episodes, time))
                break
            if len(agent.memory) > 32:
                agent.replay(32)


if __name__ == "__main__":
    main()
