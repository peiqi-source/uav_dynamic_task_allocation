"""历史版本中的tarmathmodleDQN 算法3脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


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
        self.learning_rate = 0.0005
        # model: 模型。
        self.model = self._build_model()

    def _build_model(self):
        """构建优化后的神经网络架构"""
        model = Sequential()


        model.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))


        model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))


        model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))


        model.add(Dense(self.action_size, activation='linear'))


        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, targets_hit):
        """基于epsilon贪婪策略选择动作"""
        if np.random.rand() <= self.epsilon:
            available_actions = np.where(targets_hit == 0)[0]
            return random.choice(available_actions)
        act_values = self.model.predict(state, verbose=0)
        available_actions = np.where(targets_hit == 0)[0]
        available_values = act_values[0][available_actions]
        return available_actions[np.argmax(available_values)]

    def replay(self, batch_size):
        """执行经验回放"""
        minibatch = random.sample(self.memory, batch_size)
        loss = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            loss += self.model.train_on_batch(state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss / batch_size

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


class UAVEnv:
    """UAVEnv 类，封装无人机环境相关的数据结构与业务行为。

    属性：
        data: 数据。
        num_targets: num目标集合。
        action_space: 动作space。
        state: 状态。
        targets_hit: 目标集合hit。
        defense_decay_radius: 防御能力decayradius。
        defense_decay_rate: 防御能力decay率。
        last_target_significance: last目标重要程度。
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
        # defense_decay_radius: 防御能力decayradius。
        self.defense_decay_radius = 30
        # defense_decay_rate: 防御能力decay率。
        self.defense_decay_rate = 0.1
        # last_target_significance: last目标重要程度。
        self.last_target_significance = None
        self.reset()

    def reset(self):
        """重置环境，返回初始状态"""
        self.state = self.data[0]
        self.targets_hit = np.zeros(self.num_targets)
        self.last_target_significance = None
        return self.state

    def step(self, action):
        """执行一步操作"""
        if self.targets_hit[action] == 1:
            raise ValueError("Attempting to hit an already struck target.")

        target = self.data[action]
        reward = self._calculate_reward(target)
        self.targets_hit[action] = 1


        if target[3] == 1:
            self._apply_defense_decay(action)

        done = np.all(self.targets_hit)


        if not done:
            available_actions = np.where(self.targets_hit == 0)[0]
            next_action = available_actions[np.argmax(self.data[available_actions][:, 5])]
            self.state = self.data[next_action]
        else:
            self.state = None

        return self.state, reward, done, {}

    def _apply_defense_decay(self, hit_action):
        """应用防御衰减逻辑"""
        hit_target = self.data[hit_action]
        hit_x, hit_y = hit_target[1], hit_target[2]

        for i, target in enumerate(self.data):
            if target[3] == 2 and self.targets_hit[i] == 0:
                distance = np.sqrt((target[1] - hit_x) ** 2 + (target[2] - hit_y) ** 2)
                if distance <= self.defense_decay_radius:
                    self.data[i][4] *= (1 - self.defense_decay_rate)

    def _calculate_reward(self, target):
        """根据目标特性和打击顺序计算奖励"""
        x, y, type_, defense, significance = target[1], target[2], target[3], target[4], target[5]
        reward = (significance / (defense + 1)) - 0.01 * np.sqrt(x ** 2 + y ** 2)
        if self.last_target_significance is not None:
            if significance <= self.last_target_significance:
                reward += 5
            else:
                reward -= 0.5
        self.last_target_significance = significance
        return reward


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
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        action_sequence = []
        total_loss = 0
        steps = 0

        for time in range(100):
            action = agent.act(state, env.targets_hit)
            action_sequence.append(action)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

            next_state = np.reshape(next_state, [1, state_size]) if next_state is not None else None
            if next_state is not None:
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)
                    total_loss += loss
                    steps += 1

        if steps > 0:
            avg_loss = total_loss / steps
        else:
            avg_loss = 0

        print(
            f"Episode {e + 1}/{episodes}, Total Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, Sequence: {action_sequence}")

        """if total_reward > 100:
            print("Training complete, saving model...")
            agent.save("dqn_model.h5")
            break"""

    agent.save("dqn_model.h5")
    print("Training finished.")




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
    [12, 43, 45, 1, 1, 1],
    [13, -7, 88, 1, 2, 1],
    [14, -56, 40, 1, 1, 1],
    [15, 15, 48, 2, 4, 2],
    [16, -25, 33, 2, 4, 2],
    [17, -11, 29, 1, 1, 1],
    [18, -31, 22, 1, 1, 1],
    [19, 54, 29, 2, 4, 2],
    [20, -36, 61, 2, 3, 2]
])

train_dqn(data_Target)
