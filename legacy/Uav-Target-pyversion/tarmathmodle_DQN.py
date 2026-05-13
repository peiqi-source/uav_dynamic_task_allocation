"""历史版本中的tarmathmodleDQN 算法脚本，保留用于算法对照、复现实验或迁移参考。"""
import os
import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


episode_rewards = []
losses = []

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class UAVEnv:
    """UAVEnv 类，封装无人机环境相关的数据结构与业务行为。

    属性：
        data: 数据。
        num_targets: num目标集合。
        action_space: 动作space。
        state: 状态。
        targets_hit: 目标集合hit。
    """
    def __init__(self, data_Target):
        # 说明：历史脚本沿用早期变量命名，含义请结合上下文和算法流程理解。
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            data_Target: data_Target 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        # data: 数据。
        self.data = data_Target
        # num_targets: num目标集合。
        self.num_targets = len(data_Target)
        # action_space: 动作space。
        self.action_space = np.arange(self.num_targets)
        # state: 状态。
        self.state = None
        # targets_hit: 目标集合hit。
        self.targets_hit = np.zeros(self.num_targets)
        self.reset()

    def reset(self):
        """重置环境，返回初始状态"""
        self.state = self.data[0]
        self.targets_hit = np.zeros(self.num_targets)
        return self.state

    def step(self, action):
        """执行动作，返回下一个状态、奖励、是否结束和其他信息"""
        if self.targets_hit[action] == 1:
            raise ValueError("Attempting to hit an already struck target.")

        target = self.data[action]
        reward = self._calculate_reward(target)
        self.targets_hit[action] = 1

        done = np.all(self.targets_hit)

        if not done:
            available_actions = np.where(self.targets_hit == 0)[0]
            next_action = random.choice(available_actions)
            self.state = self.data[next_action]
        else:
            self.state = None

        return self.state, reward, done, {}

    def _calculate_reward(self, target):
        """根据目标特性计算奖励"""
        x, y, type_, defense, significance = target[1], target[2], target[3], target[4], target[5]

        reward = (significance / (defense + 1)) - 0.01 * np.sqrt(x ** 2 + y ** 2)
        return reward


class DQN(tf.keras.Model):
    """DQN 类，封装DQN 算法相关的数据结构与业务行为。

    属性：
        dense1: dense1 数据。
        dense2: dense2 数据。
        output_layer: 输出layer。
    """
    def __init__(self, num_actions):
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            num_actions: num_actions 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        super(DQN, self).__init__()
        # dense1: dense1 数据。
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        # dense2: dense2 数据。
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        # output_layer: 输出layer。
        self.output_layer = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        """处理call 数据相关业务逻辑。

        参数：
            inputs: 输入集合。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


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
        target_model: 目标模型。
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
        self.memory = deque(maxlen=1000)
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
        # target_model: 目标模型。
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """构建后续流程需要的领域对象或配置对象，处理模型相关数据。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        model = DQN(self.action_size)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def update_target_model(self):
        """Update target model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store the experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, targets_hit):
        """Choose action based on epsilon-greedy policy"""
        available_actions = np.where(targets_hit == 0)[0]
        if len(available_actions) == 0:
            raise ValueError("No available targets to hit.")

        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)


        act_values = self.model(state).numpy()[0]
        act_values_filtered = np.full_like(act_values, -np.inf)
        act_values_filtered[available_actions] = act_values[available_actions]
        return np.argmax(act_values_filtered)

    def replay(self, batch_size):
        """Train the model using experience replay"""
        minibatch = random.sample(self.memory, batch_size)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

            target_f = self.model.predict(state, verbose=0)
            target_f = np.array(target_f)
            target_f[0][action] = target


            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        """将对象、模型或指标保存到指定位置。

        参数：
            name: 名称。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        self.model.save_weights(name)


def plot_metrics():
    """读取指标数据并生成可视化图表，处理指标集合相关数据。

    参数：
        无显式业务参数。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    plt.figure(figsize=(10, 5))


    plt.subplot(1, 2, 1)
    reward_line, = plt.plot([], [], label='Episode Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()


    plt.subplot(1, 2, 2)
    loss_line, = plt.plot([], [], label='Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()

    def update(frame):
        """处理update 数据相关业务逻辑。

        参数：
            frame: frame 数据。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        reward_line.set_data(range(len(episode_rewards)), episode_rewards)
        if losses:
            loss_line.set_data(range(len(losses)), losses)
        return reward_line, loss_line

    ani = animation.FuncAnimation(plt.gcf(), update, frames=None, interval=1000)
    plt.show()


def train_dqn(data_Target):
    """执行模型训练流程并保存训练产物，处理DQN 算法相关数据。

    参数：
        data_Target: 数据目标。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    state_size = 6
    action_size = len(data_Target)
    agent = DQNAgent(state_size, action_size)
    episodes = 50
    batch_size = 32


    env = UAVEnv(data_Target)

    for e in range(episodes):
        print(f"========================= Starting Episode {e + 1}/{episodes} =========================")
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(100):

            if time % 50 == 0:
                print(f"Episode {e + 1}, Step {time + 1}")

            action = agent.act(state, env.targets_hit)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward


            if done:
                print(f"Episode {e + 1} finished after {time + 1} steps, Total Reward: {total_reward}")
                break

            next_state = np.reshape(next_state, [1, state_size]) if next_state is not None else None

            if next_state is not None:
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

        if e > 0:
            print(f"Episode {e + 1}/{episodes} - Total Reward: {total_reward}")


    agent.save("dqn_model_weights.h5")


    plot_metrics()


data_Target = np.array([
    [1, -45, 28, 2, 4, 2],
    [2, 1, 74, 2, 4, 2],
    [3, 35, 42, 2, 3, 2],
    [4, 24, 46, 1, 2, 1],
    [5, -18, 38, 1, 2, 1],
    [6, 38, 21, 1, 2, 1],
    [7, -17, 66, 1, 2, 1],
    [8, -49, 44, 1, 2, 1],
    [9, 29, 7, 1, 2, 1],
    [10, 3, 96, 1, 1, 1],
    [11, -12, 15, 1, 1, 1],
    [12, 44, 66, 1, 1, 1],
    [13, 10, 60, 1, 1, 1],
    [14, -28, 84, 1, 1, 1],
    [15, -9, 51, 1, 1, 1],
    [16, 20, 80, 1, 1, 1],
    [17, -42, 25, 1, 1, 1],
    [18, 33, 46, 1, 1, 1],
    [19, 9, 32, 1, 1, 1],
    [20, -38, 70, 1, 1, 1]
])

train_dqn(data_Target)

