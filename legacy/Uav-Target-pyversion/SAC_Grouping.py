"""历史版本中的sacgrouping脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random


class ClusteringEnv:
    """ClusteringEnv 类，封装clustering环境相关的数据结构与业务行为。

    属性：
        target_data: 目标数据。
        num_clusters: num目标簇集合。
        num_targets: num目标集合。
    """
    def __init__(self, target_data, num_clusters=3):
        # 说明：历史脚本沿用早期变量命名，含义请结合上下文和算法流程理解。
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            target_data: target_data 参数。
            num_clusters: num_clusters 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        # target_data: 目标数据。
        self.target_data = target_data
        # num_clusters: num目标簇集合。
        self.num_clusters = num_clusters
        # num_targets: num目标集合。
        self.num_targets = len(target_data)
        self.reset()

    def reset(self):

        """重置对象状态，为新的回合或流程做准备。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        self.clusters = [[] for _ in range(self.num_clusters)]
        self.cluster_centers = [np.zeros(2) for _ in range(self.num_clusters)]
        return self.target_data

    def step(self, action, target_idx):
        """推进环境或仿真流程的一个时间步。

        参数：
            action: 动作。
            target_idx: 目标idx。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        target = self.target_data[target_idx]

        self.clusters[action].append(target)
        self.update_cluster_center(action)

        reward = self.calculate_compactness_reward(action) + self.calculate_balance_reward()
        done = len(sum(self.clusters, [])) == self.num_targets

        return self.target_data, reward, done

    def update_cluster_center(self, cluster_idx):

        """处理update目标簇center相关业务逻辑。

        参数：
            cluster_idx: 目标簇idx。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        points = np.array(self.clusters[cluster_idx])
        if len(points) > 0:
            self.cluster_centers[cluster_idx] = np.mean(points[:, 1:3], axis=0)

    def calculate_compactness_reward(self, cluster_idx):

        """计算指定指标或中间结果，处理compactness奖励相关数据。

        参数：
            cluster_idx: 目标簇idx。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        points = np.array(self.clusters[cluster_idx])
        if len(points) > 1:
            distances = np.linalg.norm(points[:, 1:3] - self.cluster_centers[cluster_idx], axis=1)
            compactness_reward = -np.mean(distances)
        else:
            compactness_reward = 0
        return compactness_reward

    def calculate_balance_reward(self):

        """计算指定指标或中间结果，处理balance奖励相关数据。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        sizes = [len(cluster) for cluster in self.clusters]
        if max(sizes) - min(sizes) > 1:
            balance_reward = -np.std(sizes)
        else:
            balance_reward = 0
        return balance_reward


class SACAgent:
    """SACAgent 类，封装sac智能体相关的数据结构与业务行为。

    属性：
        state_dim: 状态dim。
        action_dim: 动作dim。
        actor: actor 数据。
        critic: critic 数据。
        target_critic: 目标critic。
        actor_optimizer: actoroptimizer。
        critic_optimizer: criticoptimizer。
        gamma: 折扣因子。
        alpha: alpha 数据。
    """
    def __init__(self, state_dim, action_dim):
        # state_dim: 状态空间维度。
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            state_dim: state_dim 参数。
            action_dim: action_dim 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        # state_dim: 状态dim。
        self.state_dim = state_dim
        # action_dim: 动作dim。
        self.action_dim = action_dim
        # actor: actor 数据。
        self.actor = self.build_actor()
        # critic: critic 数据。
        self.critic = self.build_critic()
        # target_critic: 目标critic。
        self.target_critic = self.build_critic()
        # actor_optimizer: actoroptimizer。
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # critic_optimizer: criticoptimizer。
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # gamma: 折扣因子。
        self.gamma = 0.99
        # alpha: alpha 数据。
        self.alpha = 0.2

    def build_actor(self):

        """构建后续流程需要的领域对象或配置对象，处理actor 数据相关数据。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(64, activation="relu")(inputs)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(self.action_dim, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    def build_critic(self):

        """构建后续流程需要的领域对象或配置对象，处理critic 数据相关数据。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        inputs = layers.Input(shape=(self.state_dim + self.action_dim,))
        x = layers.Dense(64, activation="relu")(inputs)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(1)(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    def update(self, state, action, reward, next_state, done):
        """处理update 数据相关业务逻辑。

        参数：
            state: 状态。
            action: 动作。
            reward: 奖励。
            next_state: 下一步状态。
            done: 结束标记。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        action_one_hot = tf.one_hot([action], self.action_dim)


        with tf.GradientTape() as tape:

            target_q = reward + self.gamma * (1 - done) * self.target_critic(
                tf.concat([next_state, self.actor(next_state)], axis=1)
            )
            current_q = self.critic(tf.concat([state, action_one_hot], axis=1))
            critic_loss = tf.reduce_mean(tf.square(current_q - target_q))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))


        with tf.GradientTape() as tape:

            action_probs = self.actor(state)
            log_probs = tf.math.log(action_probs + 1e-8)
            entropy = -tf.reduce_sum(action_probs * log_probs, axis=1)
            actor_loss = tf.reduce_mean(self.critic(tf.concat([state, action_probs], axis=1)) - self.alpha * entropy)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

    def choose_action(self, state):

        """处理choose动作相关业务逻辑。

        参数：
            state: 状态。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor(state)
        action = tf.random.categorical(tf.math.log(action_probs), 1)
        return action[0, 0]


target_data = np.array([
    [1, -388, 1090, 1, 3, 2],
    [2, -479, 999, 1, 4, 2],
    [3, 156, 6, 1, 3, 2],
    [4, 724, 665, 2, 3, 2],
    [5, 982, -594, 1, 4, 2],
    [6, 885, -386, 1, 3, 2],
    [7, 1138, -1322, 2, 1, 2],
    [8, -1871, 1048, 1, 2, 2],
    [9, -943, 735, 2, 4, 2],
    [10, -405, -1017, 2, 2, 2]
])
env = ClusteringEnv(target_data)
agent = SACAgent(state_dim=target_data.shape[1], action_dim=3)


num_episodes = 500
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:

        target_idx = random.choice(range(env.num_targets))
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action, target_idx)


        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")


for idx, cluster in enumerate(env.clusters):
    print(f"Cluster {idx + 1}: {cluster}")
