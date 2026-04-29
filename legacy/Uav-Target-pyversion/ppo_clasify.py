import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd


class ClusteringEnvironment:
    def __init__(self, data_Target, num_clusters):
        self.data_Target = data_Target
        self.num_clusters = num_clusters
        self.reset()

    def reset(self):
        self.centers = self.data_Target[np.random.choice(np.where(self.data_Target[:, 3] == 2)[0], self.num_clusters), 1:3]
        return self.get_state()

    def get_state(self):
        return np.concatenate((self.centers.flatten(), self.data_Target[:, 1:3].flatten()))

    def step(self, actions):
        actions = actions.reshape(self.num_clusters, 2)  
        self.centers += actions  
        clusters = self.assign_clusters()
        reward = -self.compute_intra_cluster_distance(clusters)
        done = np.allclose(self.centers, self.centers - actions, atol=1e-3)
        return self.get_state(), reward, done, clusters

    def assign_clusters(self):
        clusters = {i: [] for i in range(self.num_clusters)}
        for i, target in enumerate(self.data_Target):
            distances = np.linalg.norm(self.centers - target[1:3], axis=1)
            assigned_cluster = np.argmin(distances)
            clusters[assigned_cluster].append(i)
        return clusters

    def compute_intra_cluster_distance(self, clusters):
        total_distance = 0
        for i, indices in clusters.items():
            if indices:
                points = self.data_Target[indices, 1:3]
                center = points.mean(axis=0)
                total_distance += np.sum(np.linalg.norm(points - center, axis=1))
        return total_distance



class SACAgent:
    def __init__(self, num_clusters, state_dim, action_dim, action_bound):
        self.num_clusters = num_clusters
        self.action_bound = action_bound

        
        self.actor = self.build_actor(state_dim, action_dim)
        self.critic1 = self.build_critic(state_dim, action_dim)
        self.critic2 = self.build_critic(state_dim, action_dim)
        self.target_critic1 = self.build_critic(state_dim, action_dim)
        self.target_critic2 = self.build_critic(state_dim, action_dim)
        self.update_target_network(self.critic1, self.target_critic1, tau=1.0)
        self.update_target_network(self.critic2, self.target_critic2, tau=1.0)

    def build_actor(self, state_dim, action_dim):
        inputs = layers.Input(shape=(state_dim,))
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)
        mu = layers.Dense(action_dim, activation="tanh")(x)
        return tf.keras.Model(inputs, mu)

    def build_critic(self, state_dim, action_dim):
        state_input = layers.Input(shape=(state_dim,))
        action_input = layers.Input(shape=(action_dim,))
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        q_value = layers.Dense(1)(x)
        return tf.keras.Model([state_input, action_input], q_value)

    def update_target_network(self, source_net, target_net, tau=0.005):
        for t, s in zip(target_net.trainable_variables, source_net.trainable_variables):
            t.assign(t * (1 - tau) + s * tau)

    def get_action(self, state):
        scaled_action = self.actor(state)
        return tf.clip_by_value(scaled_action, -self.action_bound, self.action_bound)

    
    def train(self, env, num_episodes, gamma=0.99, alpha=0.2):
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
