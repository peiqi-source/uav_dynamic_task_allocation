"""历史版本中的训练脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
from drone_battle_env import DroneBattleEnv
from dqn_model import DQNAgent

env = DroneBattleEnv()
state_space = env.observation_space
action_space = env.action_space
agent = DQNAgent(state_space, action_space)

episodes = 1000
batch_size = 32

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_space.shape[0]])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_space.shape[0]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    agent.update_target_model()
