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
    def __init__(self, data_Target):
        self.data = data_Target
        self.num_targets = len(data_Target)
        self.action_space = np.arange(self.num_targets)  
        self.state = None
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
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)  
        self.gamma = 0.95  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
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
        self.model.save_weights(name)


def plot_metrics():
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
        reward_line.set_data(range(len(episode_rewards)), episode_rewards)
        if losses:
            loss_line.set_data(range(len(losses)), losses)
        return reward_line, loss_line

    ani = animation.FuncAnimation(plt.gcf(), update, frames=None, interval=1000)
    plt.show()


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

