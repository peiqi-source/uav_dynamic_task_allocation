import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from collections import deque
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


def plot_rewards_losses(rewards):
    """绘制每轮训练的奖励曲线"""
    plt.figure(figsize=(6, 5))
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.show()


def write_to_csv(data, filename):
    """将数据写入csv文件"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Value'])
        for episode, value in enumerate(data, start=1):
            writer.writerow([episode, value])


data_Target = np.loadtxt(open('data_Target_extracted.csv'), delimiter=",", skiprows=1)

x_coords = data_Target[:, 1]
y_coords = data_Target[:, 2]

groups = {
    1: np.array([0, 1, 8, 18, 20, 36]),
    2: np.array([7, 12, 13]),
    3: np.array([22, 32, 37, 39, 44, 45, 48, 49, 50, 54, 57, 59, 60, 61]),
    4: np.array([3, 11, 19, 21]),
    5: np.array([4, 6, 27, 28]),
    6: np.array([2, 5, 16, 29, 33, 35, 38, 42, 43, 46, 51, 52]),
    7: np.array([9, 10, 14, 23, 34, 40, 41, 47, 53, 55, 56, 58]),
    8: np.array([15, 17, 24, 25, 26, 30, 31])
}


def std_x(group_indices):
    return np.std(x_coords[group_indices])


def std_y(group_indices):
    return np.std(y_coords[group_indices])


def distance_to_center(group_indices):
    center_x = np.mean(x_coords[group_indices])
    center_y = np.mean(y_coords[group_indices])
    distances = np.sqrt((x_coords[group_indices] - center_x) ** 2 + (y_coords[group_indices] - center_y) ** 2)
    return np.sum(distances)


def sum_importance_defense(group_indices):
    importance = data_Target[:, -1]
    defense = data_Target[:, -2]
    ratios = importance[group_indices] / (defense[group_indices] + 1e-10)  # 加上一个极小值避免除数为0
    return np.sum(ratios)


def get_state():
    state_features = []
    all_target_counts = []
    all_sum_imp_def_values = []
    all_dist_to_center_values = []
    all_std_x_values = []
    all_std_y_values = []
    all_group_empty_status = []

    group_lists = list(groups.values())
    for group_indices in groups.values():
        target_count = len(group_indices) if group_indices.size > 0 else 0
        all_target_counts.append(target_count)

        std_x_value = std_x(group_indices) if group_indices.size > 0 else 0
        std_y_value = std_y(group_indices) if group_indices.size > 0 else 0

        dist_to_center_value = distance_to_center(group_indices) if group_indices.size > 0 else 0
        all_dist_to_center_values.append(dist_to_center_value)

        sum_imp_def_value = sum_importance_defense(group_indices) if group_indices.size > 0 else 0
        all_sum_imp_def_values.append(sum_imp_def_value)

        all_std_x_values.append(std_x_value)
        all_std_y_values.append(std_y_value)

        group_empty_status = 1 if target_count > 0 else 0
        all_group_empty_status.append(group_empty_status)

    if all(tc == 0 for tc in all_target_counts):
        standardized_target_counts = all_target_counts
    else:
        target_count_mean = np.mean(all_target_counts)
        target_count_std = np.std(all_target_counts)
        standardized_target_counts = [(tc - target_count_mean) / target_count_std for tc in all_target_counts]

    if all(sidv == 0 for sidv in all_sum_imp_def_values):
        standardized_sum_imp_def_values = all_sum_imp_def_values
    else:
        sum_imp_def_mean = np.mean(all_sum_imp_def_values)
        sum_imp_def_std = np.std(all_sum_imp_def_values)
        standardized_sum_imp_def_values = [(sidv - sum_imp_def_mean) / sum_imp_def_std for sidv in all_sum_imp_def_values]

    if all(d == 0 for d in all_dist_to_center_values):
        normalized_dist_to_center_values = all_dist_to_center_values
    else:
        dist_to_center_min = np.min(all_dist_to_center_values)
        dist_to_center_max = np.max(all_dist_to_center_values)
        normalized_dist_to_center_values = [(d - dist_to_center_min) / (dist_to_center_max - dist_to_center_min)
                                             for d in all_dist_to_center_values]

    if all(sx == 0 for sx in all_std_x_values):
        standardized_std_x_values = all_std_x_values
    else:
        std_x_min = np.min(all_std_x_values)
        std_x_max = np.max(all_std_x_values)
        standardized_std_x_values = [(sx - std_x_min) / (std_x_max - std_x_min)
                                     for sx in all_std_x_values]

    if all(sy == 0 for sy in all_std_y_values):
        standardized_std_y_values = all_std_y_values
    else:
        std_y_min = np.min(all_std_y_values)
        std_y_max = np.max(all_std_y_values)
        standardized_std_y_values = [(sy - std_y_min) / (std_y_max - std_y_min)
                                     for sy in all_std_y_values]

    """
    standardized_group_empty_status = [
        (ges - np.min(all_group_empty_status)) / (np.max(all_group_empty_status) - np.min(all_group_empty_status)) for
        ges in all_group_empty_status]
    """

    for i in range(len(groups)):
        group_indices = group_lists[i]
        state_features.extend([standardized_target_counts[i], standardized_std_x_values[i], standardized_std_y_values[i],
                               normalized_dist_to_center_values[i], standardized_sum_imp_def_values[i],
                               all_group_empty_status[i]])

    return np.array(state_features)


action_space_size = 8 * 7


def parse_action(action):
    """将动作编号转换为源群和目标群编号"""
    source_group = action // 7 + 1
    target_group = action % 7
    if target_group == 0:
        target_group = 7
    elif target_group >= source_group:
        target_group += 1
    return source_group, target_group


def reward_function(state):
    """设计奖励函数，综合考虑目标数量均衡性、紧凑型要求和群任务量均衡性"""
    target_counts = [len(groups[i]) if groups[i].size > 0 else 0 for i in range(1, 9)]
    target_count_variance = np.var(target_counts)
    count_bonus = 1 / (1 + target_count_variance)

    avg_std_x = np.mean([std_x(groups[i]) if groups[i].size > 0 else 0 for i in range(1, 9)])
    avg_std_y = np.mean([std_y(groups[i]) if groups[i].size > 0 else 0 for i in range(1, 9)])
    compact_bonus = 1 / (1 + avg_std_x + avg_std_y)

    sum_imp_def_values = [sum_importance_defense(groups[i]) if groups[i].size > 0 else 0 for i in range(1, 9)]
    task_balance_variance = np.var(sum_imp_def_values)
    task_bonus = 1 / (1 + task_balance_variance)

    reward = 0.4 * count_bonus + 0.3 * compact_bonus + 0.3 * task_bonus
    return reward


class TargetGroupingEnv(gym.Env):
    def __init__(self):
        super(TargetGroupingEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(action_space_size)
        self.observation_space = gym.spaces.Box(low=-2, high=2, shape=(len(get_state()),), dtype=np.float32)
        self.state = get_state()

        self.reward_history = deque(maxlen=10)

    def calculate_change(self, target_index, source_group, target_group):

        original_groups = {k: v.copy() for k, v in groups.items()}

        temp_groups = self._move_target_temp(target_index, source_group, target_group)
        temp_state = get_state()

        target_counts = [len(temp_groups[i]) for i in range(1, 9)]
        target_count_variance = np.var(target_counts)

        sum_imp_def_values = [sum_importance_defense(temp_groups[i]) for i in range(1, 9)]
        task_balance_variance = np.var(sum_imp_def_values)

        variance_sum = target_count_variance + task_balance_variance

        self._restore_groups(original_groups, temp_groups)

        return variance_sum

    def step(self, action):
        print("本次选择的动作是：", action)
        source_group, target_group = parse_action(action)
        print("原群是：", source_group, "目标群是：", target_group)

        if len(groups[source_group]) == 0:
            non_empty_groups = [g for g in range(1, 9) if len(groups[g]) > 0]
            if non_empty_groups:
                source_group = np.random.choice(non_empty_groups)
                target_group = np.random.choice(range(1, 9))
                state_changes = [(idx, self.calculate_change(idx, source_group, target_group)) for idx in
                                 groups[source_group]]
                best_target_index, _ = min(state_changes, key=lambda x: x[1]) if state_changes else (None, None)
            else:
                best_target_index = None
                print("所有群都为空！")

        state_changes = []

        source_group_indices = groups[source_group]

        original_groups = {k: v.copy() for k, v in groups.items()}

        """
        print("当前各群内目标索引情况：")
        for group_num in range(1, 9):
            group_targets = groups.get(group_num, [])
            print(f"分组 {group_num}: {group_targets}")
        """

        for target_index in source_group_indices:
            temp_groups = self._move_target_temp(target_index, source_group, target_group)
            temp_state = get_state()
            temp_reward = reward_function(temp_state)

            target_counts = [len(temp_groups[i]) for i in range(1, 9)]
            target_count_variance = np.var(target_counts)

            sum_imp_def_values = [sum_importance_defense(temp_groups[i]) for i in range(1, 9)]
            task_balance_variance = np.var(sum_imp_def_values)

            variance_sum = target_count_variance + task_balance_variance
            state_changes.append((target_index, variance_sum))

            self._restore_groups(original_groups, temp_groups)

        best_target_index, _ = min(state_changes, key=lambda x: x[1])
        print("本次选择的目标是：", best_target_index)

        groups[source_group] = np.delete(groups[source_group], np.where(groups[source_group] == best_target_index))
        groups[target_group] = np.append(groups[target_group], best_target_index)

        self.state = get_state()

        print("实际移动目标后各群内目标索引情况：")
        for group_num in range(1, 9):
            group_targets = groups.get(group_num, [])
            print(f"分组 {group_num}: {group_targets}")

        reward = reward_function(self.state)
        self.reward_history.append(reward)

        done = self._check_done()

        truncated = False

        return self.state, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        global groups
        groups = {
            1: np.array([0, 1, 8, 18, 20, 36]),
            2: np.array([7, 12, 13]),
            3: np.array([22, 32, 37, 39, 44, 45, 48, 49, 50, 54, 57, 59, 60, 61]),
            4: np.array([3, 11, 19, 21]),
            5: np.array([4, 6, 27, 28]),
            6: np.array([2, 5, 16, 29, 33, 35, 38, 42, 43, 46, 51, 52]),
            7: np.array([9, 10, 14, 23, 34, 40, 41, 47, 53, 55, 56, 58]),
            8: np.array([15, 17, 24, 25, 26, 30, 31])
        }
        self.state = get_state()
        self.reward_history.clear()
        return self.state, {}

    def _move_target_temp(self, target_index, source_group, target_group):
        temp_groups = {k: v.copy() for k, v in groups.items()}
        temp_groups[source_group] = np.delete(temp_groups[source_group],
                                              np.where(temp_groups[source_group] == target_index))
        temp_groups[target_group] = np.append(temp_groups[target_group], target_index)
        return temp_groups

    def _restore_groups(self, original_groups, temp_groups):
        groups.update(original_groups)

    def _check_done(self):
        if len(self.reward_history) < self.reward_history.maxlen:
            return False

        reward_variance = np.var(self.reward_history)
        reward_stable_threshold = 1e-5

        if reward_variance < reward_stable_threshold:
            return True

        sum_imp_def_values = [sum_importance_defense(groups[i]) for i in range(1, 9)]
        task_balance_variance = np.var(sum_imp_def_values)
        task_balance_threshold = 1e-5

        if task_balance_variance < task_balance_threshold:
            return True

        return False


env = DummyVecEnv([lambda: TargetGroupingEnv()])
env.seed(None)

model = DQN("MlpPolicy", env, learning_rate=0.001, buffer_size=300, exploration_fraction=0.3,
            exploration_final_eps=0.02, verbose=1)

rewards_per_episode = []
losses_per_episode = []

num_episodes = 1000
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    step_count = 0
    while not done:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        step_count += 1
        print(f"Episode： {episode + 1}, Step： {step_count}: 当前累计奖励为： {episode_reward}")
    print(f"//////////////////////////////////////本轮结束 Episode {episode}: Reward = {episode_reward} 本轮结束//////////////////////////////////////")
    rewards_per_episode.append(episode_reward)

    """
    # 尝试从训练日志中提取损失信息
    event_acc = EventAccumulator('logs')  # 假设日志存储在'logs'目录下，根据实际调整
    event_acc.Reload()
    try:
        losses = event_acc.Scalars('loss')  # 这里的'loss'是假设的损失记录的tag，根据实际调整
        losses_per_episode.append(losses[-1].value if losses else None)  # 取最后一个时间步对应的损失值（如果有）
    except KeyError:
        losses_per_episode.append(None)
        print(f"在第 {episode + 1} 轮训练中，未找到对应的损失记录信息")
    """

    model.learn(total_timesteps=1)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

final_obs = env.reset()
final_action, _ = model.predict(final_obs, deterministic=True)
source_group, target_group = parse_action(final_action[0])
source_group_indices = groups[source_group]

groups[target_group] = np.append(groups[target_group], source_group_indices)
groups[source_group] = np.array([])

print("最终目标分群决策结果：")
for group_num in range(1, 9):
    group_targets = groups.get(group_num, [])
    print(f"分组 {group_num}: {group_targets}")

plot_rewards_losses(rewards_per_episode)
write_to_csv(rewards_per_episode, '1116_grouping_rewards.csv')