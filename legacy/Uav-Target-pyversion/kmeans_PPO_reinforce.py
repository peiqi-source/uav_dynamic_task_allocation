import gym
from gym import spaces
import numpy as np
from sklearn.cluster import KMeans
class UAVSwarmDynamicEnv(gym.Env):
    """
    自定义无人机集群动态环境
    """

    def __init__(self, num_drones=5, num_targets=10, num_clusters=3):
        super(UAVSwarmDynamicEnv, self).__init__()

        self.num_drones = num_drones
        self.num_targets = num_targets
        self.num_clusters = num_clusters

        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.num_drones + self.num_targets * 3,),
                                            dtype=np.float32)

        self.action_space = spaces.MultiDiscrete([self.num_targets] * self.num_drones)

        self.reset()

    def reset(self):
        
        self.drones = np.ones(self.num_drones, dtype=np.float32)

        self.targets = np.random.rand(self.num_targets, 3).astype(np.float32)

        self._recluster_targets()

        self.destroyed_drones = []

        self.done = False

        return self._get_obs()

    def _get_obs(self):
        
        return np.concatenate([self.drones, self.targets.flatten()])

    def step(self, actions):
        """
        actions: 一个包含每架无人机选择目标的数组
        """
        rewards = 0.0

        for drone_id, action in enumerate(actions):
            if self.drones[drone_id] <= 0:
                continue  

            if action >= self.num_targets or action < 0:
                continue  

            importance, threat, defense = self.targets[action]

            reward = importance - (threat + defense)
            rewards += reward

            self.drones[drone_id] -= 0.1  
            if self.drones[drone_id] <= 0:
                self.drones[drone_id] = 0
                self.destroyed_drones.append(drone_id)
                print(f"Drone {drone_id} destroyed!")

        self._dynamic_battlefield_changes()

        if np.all(self.drones <= 0):
            self.done = True

        return self._get_obs(), rewards, self.done, {}

    def _dynamic_battlefield_changes(self):
        """
        模拟战场动态变化：新目标出现或目标被摧毁
        """
        
        if np.random.rand() < 0.1:
            new_target = np.random.rand(1, 3).astype(np.float32)
            self.targets = np.vstack([self.targets, new_target])
            self.num_targets += 1
            self._recluster_targets()
            print("New target added!")

        
        if np.random.rand() < 0.1 and self.num_targets > 0:
            target_to_remove = np.random.randint(0, self.num_targets)
            self.targets = np.delete(self.targets, target_to_remove, axis=0)
            self.num_targets -= 1
            self._recluster_targets()
            print(f"Target {target_to_remove} removed!")

    def _recluster_targets(self):
        """
        重新聚类目标
        """
        if self.num_targets >= self.num_clusters:
            self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(self.targets)
            self.target_clusters = self.kmeans.labels_
        elif self.num_targets > 0:
            self.kmeans = KMeans(n_clusters=self.num_targets, random_state=0).fit(self.targets)
            self.target_clusters = self.kmeans.labels_
        else:
            self.target_clusters = []

    def render(self, mode='human'):
        print(f"Drones: {self.drones}")
        print(f"Targets: {self.targets}")

import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hx, cx):
        
        out, (hx, cx) = self.lstm(x, (hx, cx))  
        out = self.fc(out[:, -1, :])  
        out = self.softmax(out)
        return out, (hx, cx)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hx, cx):
        out, (hx, cx) = self.lstm(x, (hx, cx))
        out = self.fc(out[:, -1, :])
        return out, (hx, cx)


class PPOAgent:
    def __init__(self, env, gamma=0.99, eps_clip=0.2, lr=1e-4, K_epochs=4, hidden_size=128):
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.nvec[0]  

        
        self.policy = PolicyNetwork(self.input_dim, self.output_dim, hidden_size)
        self.policy_old = PolicyNetwork(self.input_dim, self.output_dim, hidden_size)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.value_net = ValueNetwork(self.input_dim, hidden_size)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])

        self.MseLoss = nn.MSELoss()
        
        self.hx = torch.zeros(1, 1, self.policy.hidden_size)
        self.cx = torch.zeros(1, 1, self.policy.hidden_size)

    def select_action(self, state, hx, cx):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  
        
        if hx is None and cx is None:
            hx = self.hx
            cx = self.cx
        probs, (hx, cx) = self.policy_old(state, hx, cx)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), hx, cx, dist.log_prob(action)

    def evaluate(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  
        action = torch.LongTensor([action]).unsqueeze(0)  
        probs, (hx, cx) = self.policy(state, None, None)
        dist = torch.distributions.Categorical(probs)

        action_logprobs = dist.log_prob(action.squeeze(-1))
        dist_entropy = dist.entropy()
        state_values, _ = self.value_net(state, None, None)

        return action_logprobs, state_values, dist_entropy

    def compute_returns_and_advantages(self, rewards, dones, values, next_value):
        returns = []
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]
        advantages = torch.FloatTensor(advantages).detach()
        returns = torch.FloatTensor(returns).detach()
        return returns, advantages

    def update(self, memory):
        
        old_states = torch.FloatTensor(memory['states'])
        old_actions = torch.LongTensor(memory['actions'])
        old_logprobs = torch.FloatTensor(memory['logprobs'])
        rewards = memory['rewards']
        dones = memory['dones']

        
        with torch.no_grad():
            state_values, _ = self.value_net(old_states.unsqueeze(1), None, None)
            state_values = state_values.squeeze()

        
        returns, advantages = self.compute_returns_and_advantages(rewards, dones, state_values, 0)

        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        
        self.policy_old.load_state_dict(self.policy.state_dict())


def train():
    env = UAVSwarmDynamicEnv(num_drones=5, num_targets=10, num_clusters=3)
    agent = PPOAgent(env)

    max_episodes = 1000
    max_timesteps = 100
    update_timestep = 2000  
    timestep = 0

    
    memory = {
        'states': [],
        'actions': [],
        'logprobs': [],
        'rewards': [],
        'dones': []
    }

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        hx = None
        cx = None
        total_reward = 0

        for t in range(max_timesteps):
            action, hx, cx, logprob = agent.select_action(state, hx, cx)
            actions = [action]  
            next_state, reward, done, _ = env.step(actions)

            
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['logprobs'].append(logprob.item())
            memory['rewards'].append(reward)
            memory['dones'].append(done)

            state = next_state
            total_reward += reward
            timestep += 1

            if timestep % update_timestep == 0:
                agent.update(memory)
                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': []}

            if done:
                break

        print(f"Episode {episode}\tReward: {total_reward}")

        
        if episode % 100 == 0:
            torch.save(agent.policy.state_dict(), f"ppo_uav_swarm_episode_{episode}.pth")


if __name__ == "__main__":
    train()
