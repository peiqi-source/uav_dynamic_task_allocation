"""历史版本中的PPO 算法clasify脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd


class ClusteringEnvironment:
    """ClusteringEnvironment 类，封装clustering环境相关的数据结构与业务行为。

    属性：
        data_Target: 数据目标。
        num_clusters: num目标簇集合。
    """
    def __init__(self, data_Target, num_clusters):
        # 说明：历史脚本沿用早期变量命名，含义请结合上下文和算法流程理解。
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            data_Target: data_Target 参数。
            num_clusters: num_clusters 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        # data_Target: 数据目标。
        self.data_Target = data_Target
        # num_clusters: num目标簇集合。
        self.num_clusters = num_clusters
        self.reset()

    def reset(self):
        """重置对象状态，为新的回合或流程做准备。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        self.centers = self.data_Target[np.random.choice(np.where(self.data_Target[:, 3] == 2)[0], self.num_clusters), 1:3]
        return self.get_state()

    def get_state(self):
        """处理get状态相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        return np.concatenate((self.centers.flatten(), self.data_Target[:, 1:3].flatten()))

    def step(self, actions):
        """推进环境或仿真流程的一个时间步。

        参数：
            actions: 动作集合。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        actions = actions.reshape(self.num_clusters, 2)
        self.centers += actions
        clusters = self.assign_clusters()
        reward = -self.compute_intra_cluster_distance(clusters)
        done = np.allclose(self.centers, self.centers - actions, atol=1e-3)
        return self.get_state(), reward, done, clusters

    def assign_clusters(self):
        """处理assign目标簇集合相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        clusters = {i: [] for i in range(self.num_clusters)}
        for i, target in enumerate(self.data_Target):
            distances = np.linalg.norm(self.centers - target[1:3], axis=1)
            assigned_cluster = np.argmin(distances)
            clusters[assigned_cluster].append(i)
        return clusters

    def compute_intra_cluster_distance(self, clusters):
        """计算指定指标或中间结果，处理intra目标簇distance相关数据。

        参数：
            clusters: 目标簇集合。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        total_distance = 0
        for i, indices in clusters.items():
            if indices:
                points = self.data_Target[indices, 1:3]
                center = points.mean(axis=0)
                total_distance += np.sum(np.linalg.norm(points - center, axis=1))
        return total_distance



class SACAgent:
    """SACAgent 类，封装sac智能体相关的数据结构与业务行为。

    属性：
        num_clusters: num目标簇集合。
        action_bound: 动作bound。
        actor: actor 数据。
        critic1: critic1 数据。
        critic2: critic2 数据。
        target_critic1: 目标critic1。
        target_critic2: 目标critic2。
    """
    def __init__(self, num_clusters, state_dim, action_dim, action_bound):
        # 说明：历史脚本沿用早期变量命名，含义请结合上下文和算法流程理解。
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            num_clusters: num_clusters 参数。
            state_dim: state_dim 参数。
            action_dim: action_dim 参数。
            action_bound: action_bound 参数。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        # num_clusters: num目标簇集合。
        self.num_clusters = num_clusters
        # action_bound: 动作bound。
        self.action_bound = action_bound


        # actor: actor 数据。
        self.actor = self.build_actor(state_dim, action_dim)
        # critic1: critic1 数据。
        self.critic1 = self.build_critic(state_dim, action_dim)
        # critic2: critic2 数据。
        self.critic2 = self.build_critic(state_dim, action_dim)
        # target_critic1: 目标critic1。
        self.target_critic1 = self.build_critic(state_dim, action_dim)
        # target_critic2: 目标critic2。
        self.target_critic2 = self.build_critic(state_dim, action_dim)
        self.update_target_network(self.critic1, self.target_critic1, tau=1.0)
        self.update_target_network(self.critic2, self.target_critic2, tau=1.0)

    def build_actor(self, state_dim, action_dim):
        """构建后续流程需要的领域对象或配置对象，处理actor 数据相关数据。

        参数：
            state_dim: 状态dim。
            action_dim: 动作dim。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        inputs = layers.Input(shape=(state_dim,))
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)
        mu = layers.Dense(action_dim, activation="tanh")(x)
        return tf.keras.Model(inputs, mu)

    def build_critic(self, state_dim, action_dim):
        """构建后续流程需要的领域对象或配置对象，处理critic 数据相关数据。

        参数：
            state_dim: 状态dim。
            action_dim: 动作dim。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        state_input = layers.Input(shape=(state_dim,))
        action_input = layers.Input(shape=(action_dim,))
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        q_value = layers.Dense(1)(x)
        return tf.keras.Model([state_input, action_input], q_value)

    def update_target_network(self, source_net, target_net, tau=0.005):
        """处理update目标神经网络相关业务逻辑。

        参数：
            source_net: sourcenet。
            target_net: 目标net。
            tau: tau 数据。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        for t, s in zip(target_net.trainable_variables, source_net.trainable_variables):
            t.assign(t * (1 - tau) + s * tau)

    def get_action(self, state):
        """处理get动作相关业务逻辑。

        参数：
            state: 状态。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        scaled_action = self.actor(state)
        return tf.clip_by_value(scaled_action, -self.action_bound, self.action_bound)


    def train(self, env, num_episodes, gamma=0.99, alpha=0.2):
        """执行模型训练流程并保存训练产物。

        参数：
            env: 环境。
            num_episodes: numepisodes。
            gamma: 折扣因子。
            alpha: alpha 数据。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
                action = self.get_action(state_tensor).numpy().flatten()

                next_state, reward, done, clusters = env.step(action)
                next_state_tensor = tf.convert_to_tensor(next_state[None, :], dtype=tf.float32)

                action_tensor = tf.convert_to_tensor(action[None, :], dtype=tf.float32)


                target_q = reward + gamma * min(
                    self.target_critic1([next_state_tensor, action_tensor]),
                    self.target_critic2([next_state_tensor, action_tensor])
                )


                current_q = min(
                    self.critic1([state_tensor, action_tensor]),
                    self.critic2([state_tensor, action_tensor])
                )


                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(tf.square(target_q - current_q))


                print(f"Current state shape: {state_tensor.shape}")
                print(f"Action shape: {action_tensor.shape}")
                print(f"Loss value: {loss.numpy()}")


                grads = tape.gradient(loss, self.actor.trainable_variables)
                for grad in grads:
                    print(grad)


                if None in grads:
                    print("Gradient is None for some variables. Check the loss computation.")

                optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

                self.update_target_network(self.critic1, self.target_critic1)
                self.update_target_network(self.critic2, self.target_critic2)

                state = next_state
                episode_reward += reward

            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
            if done:
                print("Clusters:", clusters)


df = pd.read_csv('data_Target_extracted.csv')
if df.isnull().values.any():
    print("数据包含 NaN 值，请检查数据")
data_Target = df.to_numpy()


env = ClusteringEnvironment(data_Target, num_clusters=8)
agent = SACAgent(num_clusters=8, state_dim=2 * len(data_Target) + 16, action_dim=16, action_bound=1.0)
agent.train(env, num_episodes=50)
