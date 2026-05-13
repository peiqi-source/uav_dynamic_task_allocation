"""历史版本中的攻击顺序DQN 算法脚本，保留用于算法对照、复现实验或迁移参考。"""
import os
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import csv
from tensorflow.keras import layers
import tsp_solve

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
def attack_order_dec(target_data, base):
    """
    # 目标数据和无人机基地坐标
    target_data = np.array([
       [3, 156, 6],
       [6, 885, -386],
       [17, 1167, -262],
       [30, 1135, -282],
       [34, 445, -217],
       [36, 812, 93],
       [39, 365, 176],
       [67, 543, -493],
       [68, 351, -524],
       [106, 330, -62],
       [136, 219, -86],
       [138, 275, -308]
    ])
    base_location = np.array([0, -1600])
    """
    num_targets = len(target_data)
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    learning_rate = 0.001
    num_episodes = 1000

    # 定义DQN模型
    def build_model():
       model = tf.keras.Sequential([
           layers.Dense(128, input_dim=num_targets * 2 + 2, activation='relu'),
           layers.Dense(64, activation='relu'),
           layers.Dense(num_targets, activation='linear')
       ])
       model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
       return model

    def calculate_distance(loc1, loc2):
       return np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

    def greedy_init_path(targets, base):
        path = []
        current_location = base
        visited = np.zeros(num_targets, dtype=bool)
        while not np.all(visited):
            distances = [calculate_distance(current_location, targets[i][1:]) if not visited[i] else np.inf for i in range(num_targets)]
            next_target = np.argmin(distances)
            path.append(next_target)
            visited[next_target] = True
            current_location = targets[next_target][1:]
        return path

    class UAVEnv:
       def __init__(self, targets, base):
           # targets: 目标集合。
           self.targets = targets
           # base: 基础。
           self.base = base
           self.reset()

       def reset(self):
           self.visited = np.zeros(num_targets, dtype=bool)
           self.current_location = self.base
           self.path = []
           self.total_distance = 0
           self.init_path = greedy_init_path(self.targets, self.base)
           return self._get_state()

       def _get_state(self):
           target_positions = self.targets[:, 1:].flatten()
           return np.concatenate([target_positions, self.current_location])

       def step(self, action):
           if action is None:
               return self._get_state(), 0, True

           if self.visited[action]:
               return self._get_state(), -10, False


           next_target = self.targets[action][1:]
           distance = calculate_distance(self.current_location, next_target)
           self.total_distance += distance
           self.current_location = next_target
           self.path.append(int(self.targets[action][0]))
           self.visited[action] = True

           reward = 100.0 / (distance + 1e-5)

           done = np.all(self.visited)
           if done:
               distance_back = calculate_distance(self.current_location, self.base)

               self.total_distance += distance_back
               self.path.append(self.base)

           return self._get_state(), reward, done

    # 训练DQN
    def train_dqn(env):
       global epsilon

       model = build_model()
       target_model = build_model()
       target_model.set_weights(model.get_weights())

       memory = []
       batch_size = 32
       update_target_freq = 10

       rewards = []
       losses = []

       for episode in range(num_episodes):
           state = env.reset()
           total_reward = 0
           total_loss = 0

           while True:
               if np.random.rand() < epsilon:

                   available_targets = [i for i in range(num_targets) if not env.visited[i]]
                   action = np.random.choice(available_targets) if available_targets else None
               else:
                   q_values = model.predict(state[np.newaxis], verbose=0)
                   available_targets = [i for i in range(num_targets) if not env.visited[i]]
                   if available_targets:
                       action = max(available_targets, key=lambda i: q_values[0][i])
                   else:
                       action = None
                   # action = np.argmax(q_values[0])

               next_state, reward, done = env.step(action)
               total_reward += reward

               memory.append((state, action, reward, next_state, done))
               if len(memory) > 1000:
                   memory.pop(0)

               if len(memory) >= batch_size:
                   batch = random.sample(memory, batch_size)
                   for s, a, r, ns, d in batch:
                       target = r
                       if not d:
                           target += gamma * np.amax(target_model.predict(ns[np.newaxis], verbose=0)[0])
                       q_values = model.predict(s[np.newaxis], verbose=0)
                       q_values[0][a] = target
                       loss = model.train_on_batch(s[np.newaxis], q_values)
                       total_loss += loss  # 累积损失

               if episode % update_target_freq == 0:
                   target_model.set_weights(model.get_weights())

               state = next_state
               if done:
                   break

           rewards.append(total_reward)
           losses.append(total_loss / max(len(memory), 1))  # 计算每一轮的平均损失

           if epsilon > epsilon_min:
               epsilon *= epsilon_decay

           print("Final Strike Sequence (Target Indices):", env.path)
           print(f"Episode: {episode}, Total Reward: {total_reward}, Total Loss: {total_loss}, Total Distance: {env.total_distance}")

       model.save('attack_order_dqn_model.h5')

       with open('training_rewards_losses.csv', 'w', newline='') as f:
           writer = csv.writer(f)
           writer.writerow(['Episode', 'Total Reward', 'Average Loss'])
           for i in range(num_episodes):
               writer.writerow([i, rewards[i], losses[i]])

       return model, rewards, losses

    """
    # 创建环境并训练模型
    env = UAVEnv(target_data, base_location)
    trained_model, rewards, losses = train_dqn(env)

    # 绘制奖励和损失曲线
    plt.figure(figsize=(12, 5))

    # 绘制奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.legend()

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Average Loss per Episode', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Loss per Episode')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 打印最终打击顺序
    print("Final Strike Sequence (Target Indices):")
    print(env.path)
    """

    tsp_solve.plot_strike_path(target_data, base)

