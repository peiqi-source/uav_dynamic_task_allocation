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
    def __init__(self, targets, drones, clusters):
        self.targets = targets
        self.drones = drones
        self.num_clusters = len(np.unique(clusters))
        self.clusters = clusters
        self.num_targets = len(targets)
        self.num_drones = len(drones)
        self.state = self.reset()
        self.observation_space = self.state.shape
        self.action_space = np.array([self.num_drones] * self.num_targets)

    def reset(self):
        self.state = np.zeros((self.num_targets, 2))
        return self.state

    def step(self, action):
        if not self._validate_action(action):
            reward = -float('inf')  
            done = True
            return self.state, reward, done, {}

        reward = self._compute_reward(action)
        done = self._is_done()
        self.state = self._update_state(action)
        return self.state, reward, done, {}

    def _validate_action(self, action):
        cluster_assignments = [[] for _ in range(self.num_clusters)]
        for target_id, drone_id in enumerate(action):
            cluster_id = self.clusters[target_id]
            cluster_assignments[cluster_id].append(drone_id)

        for assignments in cluster_assignments:
            if len(assignments) == 0 or not any(self.drones[drone_id] == 'recon' for drone_id in assignments):
                return False
        return True

    def _compute_reward(self, action):
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
        
        return False

    def _update_state(self, action):
        
        self.state = np.zeros((self.num_targets, 2))
        for i in range(self.num_targets):
            self.state[i, 0] = action[i]
            self.state[i, 1] = self.targets[i, 2]  
        return self.state

class DQNAgent:
    def __init__(self, state_size, action_size):
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
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.sample(range(self.action_size), self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
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
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def main():
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
