"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


STATE_DIM = 6  
ACTION_DIM = 3  


LEARNING_RATE = 0.001
GAMMA = 0.95
MEMORY_SIZE = 1000
BATCH_SIZE = 32
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01



class State:
    def __init__(self, enemy_features, uav_resources):
        self.enemy_features = enemy_features
        self.uav_resources = uav_resources

    def to_array(self):
        return np.concatenate((self.enemy_features, self.uav_resources))



class Action:
    def __init__(self, index):
        self.index = index



class ReplayMemory:
    def __init__(self):
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < MEMORY_SIZE:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % MEMORY_SIZE

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        states_array = np.array([s.to_array() for s in states])
        next_states_array = np.array([s.to_array() for s in next_states])
        return states_array, actions, rewards, next_states_array, dones



class DQN:
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=STATE_DIM))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(ACTION_DIM, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, memory):
        states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
        targets = self.model.predict(states)
        next_targets = self.target_model.predict(next_states)
        for i in range(len(states)):
            if dones[i]:
                targets[i][actions[i].index] = rewards[i]
            else:
                targets[i][actions[i].index] = rewards[i] + GAMMA * np.max(next_targets[i])
        self.model.fit(states, targets, epochs=1, batch_size=BATCH_SIZE)

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return Action(np.random.randint(ACTION_DIM))
        else:
            q_values = self.model.predict(np.array([state.to_array()]))[0]
            return Action(np.argmax(q_values))



def generate_random_state():
    enemy_features = np.random.rand(3)
    enemy_features = np.clip(enemy_features, 0, 1)
    uav_resources = np.random.rand(3)
    uav_resources = np.clip(uav_resources, 0, 1)
    return State(enemy_features, uav_resources)



def choose_random_action():
    return Action(np.random.randint(ACTION_DIM))





def simulate_environment(state, action):
    enemy_features = state.enemy_features
    uav_resources = state.uav_resources

    
    if action.index == 0:  
        uav_resources[0] += 0.1 if uav_resources[0] < 1 else 0
    elif action.index == 1:  
        uav_resources[1] += 0.1 if uav_resources[1] < 1 else 0
    elif action.index == 2:  
        uav_resources[2] += 0.1 if uav_resources[2] < 1 else 0

    
    
    
    importance = enemy_features[0]
    threat = enemy_features[1]
    defense = enemy_features[2]
    total_resource = np.sum(uav_resources)
    reward = (importance + threat + defense) * 0.5 - total_resource * 0.2 + np.random.rand()

    
    
    

    
    mean_importance = enemy_features[0]
    std_importance = 0.1
    next_importance = np.clip(np.random.normal(mean_importance, std_importance), 0, 1)

    
    mean_threat = enemy_features[1]
    std_threat = 0.1
    next_threat = np.clip(np.random.normal(mean_threat, std_threat), 0, 1)

    
    mean_defense = enemy_features[2]
    std_defense = 0.1
    next_defense = np.clip(np.random.normal(mean_defense, std_defense), 0, 1)

    next_enemy_features = np.array([next_importance, next_threat, next_defense])

    
    next_uav_resources = [r - 0.05 if r > 0 else 0 for r in uav_resources]

    next_state = State(next_enemy_features, next_uav_resources)
    done = False
    return next_state, reward, done



def train_dqn():
    dqn = DQN()
    memory = ReplayMemory()
    epsilon = EPSILON
    for episode in range(1000):
        state = generate_random_state()
        done = False
        while not done:
            action = dqn.act(state, epsilon)
            next_state, reward, done = simulate_environment(state, action)
            memory.push(state, action, reward, next_state, done)
            if len(memory.memory) > BATCH_SIZE:
                dqn.train(memory)
            state = next_state
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        if episode % 100 == 0:
            dqn.update_target_model()



if __name__ == "__main__":
    train_dqn()
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


STATE_DIM = 6  
ACTION_DIM = 3  

max_steps = 500
step_counter = 0


LEARNING_RATE = 0.001
GAMMA = 0.95
MEMORY_SIZE = 1000
BATCH_SIZE = 128
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01



class State:
    def __init__(self, enemy_features, uav_resources):
        self.enemy_features = enemy_features
        self.uav_resources = uav_resources

    def to_array(self):
        return np.concatenate((self.enemy_features, self.uav_resources))



class Action:
    def __init__(self, index):
        self.index = index



class ReplayMemory:
    def __init__(self):
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < MEMORY_SIZE:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % MEMORY_SIZE

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        states_array = np.array([s.to_array() for s in states])
        next_states_array = np.array([s.to_array() for s in next_states])
        return states_array, actions, rewards, next_states_array, dones



class DQN:
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.rewards_history = []
        self.epsilons = []

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=STATE_DIM))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(ACTION_DIM, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, memory):
        states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
        targets = self.model.predict(states)
        next_targets = self.target_model.predict(next_states)
        for i in range(len(states)):
            if dones[i]:
                targets[i][actions[i].index] = rewards[i]
            else:
                targets[i][actions[i].index] = rewards[i] + GAMMA * np.max(next_targets[i])
        self.model.fit(states, targets, epochs=1, batch_size=BATCH_SIZE)
        self.rewards_history.append(np.mean(rewards))

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return Action(np.random.randint(ACTION_DIM))
        else:
            q_values = self.model.predict(np.array([state.to_array()]))[0]
            return Action(np.argmax(q_values))



def generate_random_state():
    enemy_features = np.random.rand(3)
    enemy_features = np.clip(enemy_features, 0, 1)
    uav_resources = np.random.rand(3)
    uav_resources = np.clip(uav_resources, 0, 1)
    return State(enemy_features, uav_resources)



def choose_random_action():
    return Action(np.random.randint(ACTION_DIM))





def simulate_environment(state, action):
    global step_counter
    enemy_features = state.enemy_features
    uav_resources = state.uav_resources

    
    if action.index == 0:  
        uav_resources[0] += 0.1 if uav_resources[0] < 1 else 0
    elif action.index == 1:  
        uav_resources[1] += 0.1 if uav_resources[1] < 1 else 0
    elif action.index == 2:  
        uav_resources[2] += 0.1 if uav_resources[2] < 1 else 0

    
    
    
    importance = enemy_features[0]
    threat = enemy_features[1]
    defense = enemy_features[2]
    total_resource = np.sum(uav_resources)
    reward = (importance + threat + defense) * 0.5 - total_resource * 0.2 + np.random.rand()

    
    
    

    
    mean_importance = enemy_features[0]
    std_importance = 0.1
    next_importance = np.clip(np.random.normal(mean_importance, std_importance), 0, 1)

    
    mean_threat = enemy_features[1]
    std_threat = 0.1
    next_threat = np.clip(np.random.normal(mean_threat, std_threat), 0, 1)

    
    mean_defense = enemy_features[2]
    std_defense = 0.1
    next_defense = np.clip(np.random.normal(mean_defense, std_defense), 0, 1)

    next_enemy_features = np.array([next_importance, next_threat, next_defense])

    
    next_uav_resources = [r - 0.05 if r > 0 else 0 for r in uav_resources]

    step_counter += 1
    if step_counter >= max_steps:
        done = True
    else:
        done = False

    next_state = State(next_enemy_features, next_uav_resources)
    return next_state, reward, done



def train_dqn():
    dqn = DQN()
    memory = ReplayMemory()
    epsilon = EPSILON
    for episode in range(200):
        state = generate_random_state()
        done = False
        while not done:
            action = dqn.act(state, epsilon)
            next_state, reward, done = simulate_environment(state, action)
            memory.push(state, action, reward, next_state, done)
            if len(memory.memory) > BATCH_SIZE:
                dqn.train(memory)
            state = next_state
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        dqn.epsilons.append(epsilon)
        if episode % 100 == 0:
            dqn.update_target_model()

    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(dqn.rewards_history)
    plt.title('Average Reward over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')

    plt.subplot(1, 2, 2)
    plt.plot(dqn.epsilons)
    plt.title('Epsilon Decay over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')

    plt.show()

    return dqn



if __name__ == "__main__":
    trained_dqn = train_dqn()

    
    total_reward = 0
    num_episodes = 10
    for _ in range(num_episodes):
        state = generate_random_state()
        done = False
        while not done:
            action = trained_dqn.act(state, 0.0)  
            next_state, reward, done = simulate_environment(state, action)
            total_reward += reward
            state = next_state
    average_reward = total_reward / num_episodes
    print(f'Average reward over {num_episodes} evaluation episodes: {average_reward}')

