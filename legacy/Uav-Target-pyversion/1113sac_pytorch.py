import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 超参数
learning_rate = 0.001
gamma = 0.99
alpha = 0.2
w1 = 0.5
w2 = 0.5
batch_size = 64
max_episodes = 1000
memory_size = 10000
epsilon = 0.1

groups = {
    1: {'target_num': 6, 'indices': [0, 1, 8, 18, 20, 36]},
    2: {'target_num': 3, 'indices': [7, 12, 13]},
    3: {'target_num': 14, 'indices': [22, 32, 37, 39, 44, 45, 48, 49, 50, 54, 57, 59, 60, 61]},
    4: {'target_num': 4, 'indices': [3, 11, 19, 21]},
    5: {'target_num': 4, 'indices': [4, 6, 27, 28]},
    6: {'target_num': 12, 'indices': [2, 5, 16, 29, 33, 35, 38, 42, 43, 46, 51, 52]},
    7: {'target_num': 12, 'indices': [9, 10, 14, 23, 34, 40, 41, 47, 53, 55, 56, 58]},
    8: {'target_num': 7, 'indices': [15, 17, 24, 25, 26, 30, 31]}
}

data_Target = np.loadtxt(open('data_Target_extracted.csv'), delimiter=",", skiprows=1)
print("目标总计：", len(data_Target))

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        return action_probs


# 定义价值网络
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        state_value = self.value_head(x)
        return state_value

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        assert isinstance(action, int) and 0 <= action < len(action_space), "Invalid action in push"
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        next_state = np.array(next_state)
        print("Sampled actions from buffer:", action)
        return torch.FloatTensor(state), torch.LongTensor(action), torch.FloatTensor(reward), torch.FloatTensor(
            next_state)

    def __len__(self):
        return len(self.buffer)

def calculate_reward_balance(groups):
    avg_num = sum([g['target_num'] for g in groups.values()]) / len(groups)
    reward_balance = -sum([(g['target_num'] - avg_num) ** 2 for g in groups.values()])
    return reward_balance

def calculate_reward_compactness(groups, positions):
    reward_compact = 0
    for group in groups.values():
        group_positions = positions[np.array(group['indices'])]
        variance_x = np.var(group_positions[:, 0])
        variance_y = np.var(group_positions[:, 1])
        reward_compact += 1 / (variance_x + variance_y + 1e-6)
    return reward_compact

def get_state(groups, positions):
    state = []
    avg_num = sum([g['target_num'] for g in groups.values()]) / len(groups)
    for group in groups.values():
        group_positions = positions[np.array(group['indices'])]

        target_num_diff = (group['target_num'] - avg_num)
        min_target_num_diff = min([g['target_num'] - avg_num for g in groups.values()])
        max_target_num_diff = max([g['target_num'] - avg_num for g in groups.values()])
        if max_target_num_diff == min_target_num_diff:
            norm_target_num_diff = 0
        else:
            norm_target_num_diff = (target_num_diff - min_target_num_diff) / (max_target_num_diff - min_target_num_diff)
        state.append(norm_target_num_diff)

        avg_x = np.mean(group_positions[:, 0]) if group['indices'] else 0
        min_avg_x = np.min([np.mean(positions[np.array(g['indices'])][:, 0]) for g in groups.values() if g['indices']])
        max_avg_x = np.max([np.mean(positions[np.array(g['indices'])][:, 0]) for g in groups.values() if g['indices']])
        if max_avg_x == min_avg_x:
            norm_avg_x = 0
        else:
            norm_avg_x = (avg_x - min_avg_x) / (max_avg_x - min_avg_x)
        state.append(norm_avg_x)

        avg_y = np.mean(group_positions[:, 1]) if group['indices'] else 0
        min_avg_y = np.min([np.mean(positions[np.array(g['indices'])][:, 1]) for g in groups.values() if g['indices']])
        max_avg_y = np.max([np.mean(positions[np.array(g['indices'])][:, 1]) for g in groups.values() if g['indices']])
        if max_avg_y == min_avg_y:
            norm_avg_y = 0
        else:
            norm_avg_y = (avg_y - min_avg_y) / (max_avg_y - min_avg_y)
        state.append(norm_avg_y)

        std_x = np.std(group_positions[:, 0]) if group['indices'] else 0
        min_std_x = np.min([np.std(positions[np.array(g['indices'])][:, 0]) for g in groups.values() if g['indices']])
        max_std_x = np.max([np.std(positions[np.array(g['indices'])][:, 0]) for g in groups.values() if g['indices']])
        if max_std_x == min_std_x:
            norm_std_x = 0
        else:
            norm_std_x = (std_x - min_std_x) / (max_std_x - min_std_x)
        state.append(norm_std_x)

        std_y = np.std(group_positions[:, 1]) if group['indices'] else 0
        min_std_y = np.min([np.std(positions[np.array(g['indices'])][:, 1]) for g in groups.values() if g['indices']])
        max_std_y = np.max([np.std(positions[np.array(g['indices'])][:, 1]) for g in groups.values() if g['indices']])
        if max_std_y == min_std_y:
            norm_std_y = 0
        else:
            norm_std_y = (std_y - min_std_y) / (max_std_y - min_std_y)
        state.append(norm_std_y)

    return np.array(state).astype(np.float32)

def build_action_space(groups):
    action_space = []
    for target_idx in range(1, 63):
        current_group = None
        for group_id, group_info in groups.items():
            if target_idx in group_info['indices']:
                current_group = group_id
                break
        for target_group in range(1, 9):
            if target_group!= current_group:
                action_space.append((current_group, target_group, target_idx))
    return action_space

action_space = build_action_space(groups)
action_dim = len(action_space)
print("action_dim:", action_dim)
state_dim = 5 * 8

policy_net = PolicyNetwork(state_dim, action_dim)
value_net = ValueNetwork(state_dim)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)

memory = ReplayBuffer(memory_size)

# 训练
for episode in range(max_episodes):
    state = get_state(groups, data_Target[:, 1:3])
    episode_reward = 0
    done = False
    while not done:

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy_net(state_tensor).squeeze(0)
        print("Action probabilities:", action_probs)
        result = action_probs.sum().item()
        print("Action probabilities result:", result)
        if random.random() < epsilon:
            action_idx = random.randrange(action_dim)
            assert action_idx < len(action_space), "Random action index out of bounds"
        else:
            action_idx = torch.multinomial(action_probs, 1).item()
            assert 0 <= action_idx < len(action_space), "Sampled action index out of bounds"
        action = action_space[action_idx]


        source_group = action[0]
        target_group = action[1]
        target_index = action[2]
        if target_index in groups[source_group]['indices']:
            groups[source_group]['indices'].remove(target_index)
            groups[source_group]['target_num'] -= 1
            groups[target_group]['indices'].append(target_index)
            groups[target_group]['target_num'] += 1

        next_state = get_state(groups, data_Target[:, 1:3])
        reward_balance = calculate_reward_balance(groups)
        reward_compact = calculate_reward_compactness(groups, data_Target[:, 1:3])
        reward = w1 * reward_balance + w2 * reward_compact
        episode_reward += reward

        memory.push(state, action_idx, reward, next_state)

        if len(memory) >= batch_size:
            states, actions, rewards, next_states = memory.sample(batch_size)

            with torch.no_grad():
                next_state_values = value_net(next_states).squeeze(-1)
            target_values = rewards + gamma * next_state_values
            value_preds = value_net(states).squeeze(-1)
            value_loss = nn.MSELoss()(value_preds, target_values)

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            print("actions shape:", actions.shape)
            print("actions values:", actions)
            print("value_net(states) shape:", value_net(states).shape)
            action_dim = len(action_space)
            for idx in actions:
                assert 0 <= idx < action_dim, f"Action index {idx} out of bounds. Valid range is [0, {action_dim - 1}]"

            state_action_values = value_net(states).gather(1, actions.unsqueeze(1)).squeeze(-1)
            log_action_probs = torch.log(policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(-1))
            policy_loss = (alpha * log_action_probs - state_action_values).mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

        state = next_state
    print(f'Episode {episode}: Reward = {episode_reward}')

for group_id, group_info in groups.items():
    print(f'Group {group_id}: Target number = {group_info["target_num"]}, Indices = {group_info["indices"]}')
