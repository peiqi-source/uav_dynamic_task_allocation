"""历史版本中的tarmathmodleDQN 算法2脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from torch.utils.tensorboard import SummaryWriter


class DQNAgent:
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):

        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, targets_hit):

        if np.random.rand() <= self.epsilon:
            available_actions = np.where(targets_hit == 0)[0]
            return random.choice(available_actions)
        act_values = self.model.predict(state, verbose=0)
        available_actions = np.where(targets_hit == 0)[0]
        available_values = act_values[0][available_actions]
        return available_actions[np.argmax(available_values)]

    def replay(self, batch_size):

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
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class UAVEnv:
        self.data = data_Target
        self.num_targets = len(data_Target)
        self.action_space = np.arange(self.num_targets)
        self.state = None
        self.targets_hit = np.zeros(self.num_targets)
        self.defense_decay_radius = 30
        self.defense_decay_rate = 0.1
        self.last_target_significance = None
        self.reset()

    def reset(self):
        """重置环境，返回初始状态"""
        self.state = self.data[0]
        self.targets_hit = np.zeros(self.num_targets)
        self.last_target_significance = None
        return self.state

    def step(self, action):
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

        """
        for other_target in self.data:
            if other_target[3] == 2 and self.targets_hit[np.where(self.data == other_target)[0][0]] == 0:
                distance = np.sqrt((other_target[1] - x) ** 2 + (other_target[2] - y) ** 2)
                if distance < self.defense_decay_radius:
                    reward -= 1

        return reward


def train_dqn(data_Target):
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

        action_sequence = []
        total_loss = 0
        steps = 0

        for time in range(100):

            """if time % 50 == 0:
                print(f"Episode {e + 1}, Step {time + 1}")"""

            action = agent.act(state, env.targets_hit)

            action_sequence.append(action)
            next_state, reward, done, _ = env.step(action)

            """
            print(f"状态转移检查Episode {e + 1}, Step {time + 1}")
            print(f"当前状态: {state}")
            print(f"执行动作: {action}")
            print(f"下一状态: {next_state}")
            print(f"奖励: {reward}")


            print(f"执行动作: {action}, 击中的目标: {data_Target[action]}")
            print(f"目标防御值（打击前）: {data_Target[action][4]}")


            if data_Target[action][3] == 1:
                print("检查周围type=2目标的防御衰减...")
                for i, target in enumerate(data_Target):

                    distance = np.sqrt(
                        (target[1] - data_Target[action][1]) ** 2 + (target[2] - data_Target[action][2]) ** 2)
                    if target[3] == 2 and distance < 50:
                        print(f"目标 {i} 的防御值已衰减: {target[4]}")"""

            total_reward += reward


            if done:
                print(f"Episode {e + 1} finished after {time + 1} steps, Total Reward: {total_reward}")
                break

            next_state = np.reshape(next_state, [1, state_size]) if next_state is not None else None

            if next_state is not None:
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)

                    total_loss += loss
                    steps += 1

        if e > 0:
            print(f"Episode {e + 1}/{episodes} - Total Reward: {total_reward}")
        if steps > 0:
            average_loss = total_loss / steps
            print(f"Episode {e + 1} Average Loss: {average_loss:.4f}")


        print(f"Episode {e + 1} Action Sequence: {action_sequence}")


    agent.save("dqn_model_weights.h5")


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
