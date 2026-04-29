import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


num_episodes = 1000
gamma = 0.99  
alpha = 0.2  

def initialize_state(num_targets):
    """
    初始化状态空间
    :param num_targets: 目标的数量
    :return: 初始化后的状态字典
    """
    state = {
        "positions": np.zeros((num_targets, 2)),  
        "distance_matrix": np.zeros((num_targets, num_targets)),  
        "cluster_assignments": np.full(num_targets, -1),  
        "cluster_sizes": np.zeros(8)  
    }
    return state


def build_policy_network(state_dim, action_dim):
    """
    构建策略网络
    :param state_dim: 状态空间维度
    :param action_dim: 动作空间维度（8个目标群，所以为8）
    :return: 构建好的策略网络模型
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(state_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(action_dim, activation='softmax')
    ])
    return model


def build_value_network(state_dim):
    """
    构建价值网络
    :param state_dim: 状态空间维度
    :param action_dim: 此处不用，但为了接口统一保留参数
    :return: 构建好的价值网络模型
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(state_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

data_Target_extracted = np.array([[1, -388, 1090, 1, 3, 2], [2, -479, 999, 1, 4, 2], [3, 156, 6, 1, 3, 2], [4, 724, 665, 2, 3, 2],
                                  [5, 982, -594, 1, 4, 2], [6, 885, -386, 1, 3, 2], [7, 1138, -1322, 2, 1, 2], [8, -1871, 1048, 1, 2, 2],
                                  [9, -943, 735, 2, 4, 2], [10, -405, -1017, 2, 2, 2], [11, 740, -1411, 2, 4, 2], [12, 1453, 1077, 2, 4, 2],
                                  [13, -1413, 93, 2, 2, 2], [14, -1942, 149, 2, 3, 2], [15, 491, -1114, 2, 4, 2], [16, -1794, -1253, 2, 1, 2],
                                  [17, 1167, -262, 2, 2, 2], [18, -1575, -1288, 2, 3, 2], [19, -647, 239, 2, 2, 2], [20, 330, 1120, 2, 2, 2],
                                  [21, -687, 658, 2, 1, 2], [22, 523, 1349, 2, 4, 2], [23, -330, -108, 2, 1, 2], [24, 297, -853, 2, 3, 2],
                                  [25, -1945, -1188, 2, 4, 2], [26, -1743, -712, 2, 3, 2], [27, -1957, -822, 2, 2, 2], [28, 1959, -707, 2, 1, 2],
                                  [29, 1655, -992, 2, 1, 2], [30, 1135, -282, 2, 4, 2], [31, -1368, -1439, 2, 4, 2], [32, -1494, -625, 2, 4, 2],
                                  [33, 37, -400, 2, 1, 2], [34, 445, -217, 2, 1, 2], [35, 483, -925, 2, 3, 2], [36, 812, 93, 2, 4, 2],
                                  [37, 9, 557, 2, 1, 2], [38, -657, 74, 2, 4, 2], [39, 365, 176, 2, 1, 2], [40, 73, -189, 2, 1, 2],
                                  [43, 18, -1191, 1, 3, 1], [57, 132, -735, 1, 4, 1], [67, 543, -493, 1, 3, 1], [68, 351, -524, 1, 3, 1],
                                  [89, -390, -130, 1, 2, 1], [104, -308, -481, 1, 3, 1], [106, 330, -62, 1, 4, 1], [119, 167, -1169, 1, 2, 1],
                                  [123, -689, -443, 1, 1, 1], [124, -251, -209, 1, 1, 1], [128, -361, -501, 1, 1, 1], [136, 219, -86, 1, 1, 1],
                                  [138, 275, -308, 1, 1, 1], [141, -342, -1028, 1, 4, 1], [147, 60, -70, 1, 4, 1], [149, 402, -697, 1, 2, 1],
                                  [151, 370, -772, 1, 3, 1], [158, -631, -225, 1, 2, 1], [167, 199, -1097, 1, 3, 1], [171, -671, -381, 1, 1, 1],
                                  [174, 4, -9, 1, 3, 1], [196, 86, -264, 1, 3, 1]])

num_targets = len(data_Target_extracted)  
state_dim = sum([v.size for v in initialize_state(num_targets).values()])  
action_dim = 8

policy_net = build_policy_network(state_dim, action_dim)
value_net = build_value_network(state_dim)

optimizer_policy = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer_value = tf.keras.optimizers.Adam(learning_rate=0.001)


def calculate_distance_matrix(positions):
    """
    计算目标之间的距离矩阵
    :param positions: 目标位置坐标数组，形状为 (num_targets, 2)
    :return: 距离矩阵，形状为 (num_targets, num_targets)
    """
    num_targets = positions.shape[0]
    distance_matrix = np.zeros((num_targets, num_targets))
    for i in range(num_targets):
        for j in range(num_targets):
            if i!= j:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance_matrix[i, j] = np.sqrt(dx ** 2 + dy ** 2)
    return distance_matrix


def update_cluster_assignments(state, action, target_index):
    """
    根据动作更新目标所属群的分配情况
    :param state: 当前状态字典
    :param action: 采取的动作（群编号，范围 0 - 7）
    :param target_index: 要分配的目标索引
    :return: 更新后的状态字典
    """
    state["cluster_assignments"][target_index] = action
    state["cluster_sizes"][action] += 1
    return state


def update_distance_matrix(state):
    """
    根据最新的目标分群情况更新距离矩阵（考虑群内目标距离）
    :param state: 当前状态字典
    :return: 更新后的状态字典
    """
    positions = state["positions"]
    cluster_assignments = state["cluster_assignments"]
    num_targets = len(cluster_assignments)
    new_distance_matrix = np.zeros((num_targets, num_targets))
    for i in range(num_targets):
        for j in range(num_targets):
            if cluster_assignments[i] == cluster_assignments[j]:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                new_distance_matrix[i, j] = np.sqrt(dx ** 2 + dy ** 2)
    state["distance_matrix"] = new_distance_matrix
    return state


def take_action(state, action):
    """
    执行动作，更新状态，获取奖励以及判断是否结束
    :param state: 当前状态字典
    :param action: 采取的动作（群编号，范围 0 - 7）
    :return: 下一个状态字典，奖励值，是否结束的布尔值（这里暂未实际使用结束条件，始终返回False）
    """
    
    unassigned_target_indices = np.where(state["cluster_assignments"] == -1)[0]
    if len(unassigned_target_indices) > 0:
        target_index = unassigned_target_indices[0]
        
        next_state = update_cluster_assignments(state, action, target_index)
        
        next_state = update_distance_matrix(next_state)
        
        reward = total_reward(next_state)
        return next_state, reward, False
    else:
        return state, 0, True  


def compactness_reward(cluster_assignments, positions):
    """
    计算位置紧凑性奖励
    :param cluster_assignments: 目标所属群编号数组
    :param positions: 目标位置坐标数组
    :return: 位置紧凑性奖励值
    """
    rewards = []
    for cluster_id in range(8):
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        if len(cluster_indices) > 0:
            cluster_positions = positions[cluster_indices]
            std_x = np.std(cluster_positions[:, 0])
            std_y = np.std(cluster_positions[:, 1])
            std_total = std_x + std_y
            rewards.append(1 / (1 + std_total))
    return np.mean(rewards) if rewards else 0


def balance_reward(cluster_sizes):
    """
    计算目标数量均衡性奖励
    :param cluster_sizes: 每个群包含的目标数量数组
    :return: 目标数量均衡性奖励值
    """
    max_size = np.max(cluster_sizes)
    min_size = np.min(cluster_sizes)
    diff = max_size - min_size
    return 1 / (1 + diff) if diff >= 0 else 1


def total_reward(state):
    """
    计算综合奖励，是位置紧凑性奖励和目标数量均衡性奖励的加权和
    :param state: 当前状态字典
    :return: 综合奖励值
    """
    compactness_r = compactness_reward(state["cluster_assignments"], state["positions"])
    balance_r = balance_reward(state["cluster_sizes"])
    return 0.5 * compactness_r + 0.5 * balance_r


for episode in range(num_episodes):
    state = initialize_state(num_targets)
    state["positions"] = data_Target_extracted[:, 1:3]  
    done = False
    while not done:
        
        positions = state["positions"]
        distance_matrix = state["distance_matrix"]
        cluster_assignments = state["cluster_assignments"]
        cluster_sizes = state["cluster_sizes"]

        
        positions_tensor = tf.convert_to_tensor(np.array([positions]), dtype=tf.float32)
        
        distance_matrix_tensor = tf.convert_to_tensor(np.array([distance_matrix[:, :, 0:2]]), dtype=tf.float32)
        cluster_assignments_tensor = tf.convert_to_tensor(np.array([cluster_assignments]), dtype=tf.float32)
        cluster_sizes_tensor = tf.convert_to_tensor(np.array([cluster_sizes]), dtype=tf.float32)

        
        state_flatten = tf.concat(
            [positions_tensor, distance_matrix_tensor, cluster_assignments_tensor, cluster_sizes_tensor], axis=1)
        with tf.GradientTape() as tape_policy:
            action_probs = policy_net(state_flatten)
            action_dist = tfp.distributions.Categorical(probs=action_probs)
            action = action_dist.sample().numpy()[0]  

            
            next_state, reward, done = take_action(state, action)

            target_value = reward + gamma * value_net(tf.convert_to_tensor(np.array([next_state]).reshape(1, -1),
                                                                         dtype=tf.float32))
            log_probs = action_dist.log_prob(action)
            policy_loss = (alpha * log_probs - value_net(state_flatten))

            
            policy_loss = tf.reduce_mean(policy_loss)

        policy_gradients = tape_policy.gradient(policy_loss, policy_net.trainable_variables)
        optimizer_policy.apply_gradients(zip(policy_gradients, policy_net.trainable_variables))

        with tf.GradientTape() as tape_value:
            value_loss = tf.keras.losses.MeanSquaredError()(
                value_net(state_flatten), target_value)

        value_gradients = tape_value.gradient(value_loss, value_net.trainable_variables)
        optimizer_value.apply_gradients(zip(value_gradients, value_net.trainable_variables))

        state = next_state

    if episode % 100 == 0:
        print(f'Episode {episode}: Reward = {reward}')