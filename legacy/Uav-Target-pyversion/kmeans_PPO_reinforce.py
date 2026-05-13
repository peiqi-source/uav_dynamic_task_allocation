"""历史版本中的KMeans 算法PPO 算法reinforce脚本，保留用于算法对照、复现实验或迁移参考。"""
import gym
from gym import spaces
import numpy as np
from sklearn.cluster import KMeans
class UAVSwarmDynamicEnv(gym.Env):
    """
    自定义无人机集群动态环境
    """

    def __init__(self, num_drones=5, num_targets=10, num_clusters=3):
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            num_drones: num_drones 参数。
            num_targets: num_targets 参数。
            num_clusters: num_clusters 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        super(UAVSwarmDynamicEnv, self).__init__()

        # num_drones: numdrones。
        self.num_drones = num_drones
        # num_targets: num目标集合。
        self.num_targets = num_targets
        # num_clusters: num目标簇集合。
        self.num_clusters = num_clusters

        # observation_space: 观测向量space。
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.num_drones + self.num_targets * 3,),
                                            dtype=np.float32)

        # action_space: 动作space。
        self.action_space = spaces.MultiDiscrete([self.num_targets] * self.num_drones)

        self.reset()

    def reset(self):

        """重置对象状态，为新的回合或流程做准备。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        self.drones = np.ones(self.num_drones, dtype=np.float32)

        self.targets = np.random.rand(self.num_targets, 3).astype(np.float32)

        self._recluster_targets()

        self.destroyed_drones = []

        self.done = False

        return self._get_obs()

    def _get_obs(self):

        """处理getobs相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
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
        """处理render 数据相关业务逻辑。

        参数：
            mode: mode 数据。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        print(f"Drones: {self.drones}")
        print(f"Targets: {self.targets}")

import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """PolicyNetwork 类，封装策略神经网络相关的数据结构与业务行为。

    属性：
        hidden_size: hiddensize。
        lstm: lstm 数据。
        fc: fc 数据。
        softmax: softmax 数据。
    """
    def __init__(self, input_dim, output_dim, hidden_size=128):
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            input_dim: input_dim 参数。
            output_dim: output_dim 参数。
            hidden_size: hidden_size 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        super(PolicyNetwork, self).__init__()
        # hidden_size: hiddensize。
        self.hidden_size = hidden_size
        # lstm: lstm 数据。
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        # fc: fc 数据。
        self.fc = nn.Linear(hidden_size, output_dim)
        # softmax: softmax 数据。
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hx, cx):

        """处理forward 数据相关业务逻辑。

        参数：
            x: 横坐标。
            hx: hx 数据。
            cx: cx 数据。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        out, (hx, cx) = self.lstm(x, (hx, cx))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out, (hx, cx)


class ValueNetwork(nn.Module):
    """ValueNetwork 类，封装数值神经网络相关的数据结构与业务行为。

    属性：
        hidden_size: hiddensize。
        lstm: lstm 数据。
        fc: fc 数据。
    """
    def __init__(self, input_dim, hidden_size=128):
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            input_dim: input_dim 参数。
            hidden_size: hidden_size 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        super(ValueNetwork, self).__init__()
        # hidden_size: hiddensize。
        self.hidden_size = hidden_size
        # lstm: lstm 数据。
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        # fc: fc 数据。
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hx, cx):
        """处理forward 数据相关业务逻辑。

        参数：
            x: 横坐标。
            hx: hx 数据。
            cx: cx 数据。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        out, (hx, cx) = self.lstm(x, (hx, cx))
        out = self.fc(out[:, -1, :])
        return out, (hx, cx)


class PPOAgent:
    """PPOAgent 类，封装PPO 算法智能体相关的数据结构与业务行为。

    属性：
        env: 环境。
        gamma: 折扣因子。
        eps_clip: epsclip。
        K_epochs: kepochs。
        input_dim: 输入dim。
        output_dim: 输出dim。
        policy: 策略。
        policy_old: 策略old。
        value_net: 数值net。
        optimizer: optimizer 数据。
        MseLoss: mse损失值。
        hx: hx 数据。
        cx: cx 数据。
    """
    def __init__(self, env, gamma=0.99, eps_clip=0.2, lr=1e-4, K_epochs=4, hidden_size=128):
        # 说明：历史脚本沿用早期变量命名，含义请结合上下文和算法流程理解。
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            env: 环境对象。
            gamma: gamma 参数。
            eps_clip: eps_clip 参数。
            lr: lr 参数。
            K_epochs: K_epochs 参数。
            hidden_size: hidden_size 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        # env: 环境。
        self.env = env
        # gamma: 折扣因子。
        self.gamma = gamma
        # eps_clip: epsclip。
        self.eps_clip = eps_clip
        # K_epochs: kepochs。
        self.K_epochs = K_epochs

        # input_dim: 输入dim。
        self.input_dim = env.observation_space.shape[0]
        # output_dim: 输出dim。
        self.output_dim = env.action_space.nvec[0]


        # policy: 策略。
        self.policy = PolicyNetwork(self.input_dim, self.output_dim, hidden_size)
        # policy_old: 策略old。
        self.policy_old = PolicyNetwork(self.input_dim, self.output_dim, hidden_size)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # value_net: 数值net。
        self.value_net = ValueNetwork(self.input_dim, hidden_size)

        # optimizer: optimizer 数据。
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])

        # MseLoss: mse损失值。
        self.MseLoss = nn.MSELoss()

        # hx: hx 数据。
        self.hx = torch.zeros(1, 1, self.policy.hidden_size)
        # cx: cx 数据。
        self.cx = torch.zeros(1, 1, self.policy.hidden_size)

    def select_action(self, state, hx, cx):
        """按照策略从候选集合中选择目标对象，处理动作相关数据。

        参数：
            state: 状态。
            hx: hx 数据。
            cx: cx 数据。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)

        if hx is None and cx is None:
            hx = self.hx
            cx = self.cx
        probs, (hx, cx) = self.policy_old(state, hx, cx)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), hx, cx, dist.log_prob(action)

    def evaluate(self, state, action):
        """执行评估流程并输出评估指标。

        参数：
            state: 状态。
            action: 动作。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        action = torch.LongTensor([action]).unsqueeze(0)
        probs, (hx, cx) = self.policy(state, None, None)
        dist = torch.distributions.Categorical(probs)

        action_logprobs = dist.log_prob(action.squeeze(-1))
        dist_entropy = dist.entropy()
        state_values, _ = self.value_net(state, None, None)

        return action_logprobs, state_values, dist_entropy

    def compute_returns_and_advantages(self, rewards, dones, values, next_value):
        """计算指定指标或中间结果，处理returnsandadvantages相关数据。

        参数：
            rewards: rewards 数据。
            dones: dones 数据。
            values: values 数据。
            next_value: 下一步数值。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
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

        """处理update 数据相关业务逻辑。

        参数：
            memory: memory 数据。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
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
    """执行模型训练流程并保存训练产物。

    参数：
        无显式业务参数。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
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
