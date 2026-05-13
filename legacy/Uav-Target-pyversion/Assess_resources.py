"""历史版本中的assess资源集合脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np

target_data = np.array([
    [1, 100, 5, 50, 10, 20],
    [2, 150, 3, 70, 15, 25],
    [3, 120, 10, 40, 20, 30],
    [4, 90, 8, 60, 25, 35]
], dtype=float)

def distance_decay_factor(distance):
    """处理distancedecayfactor相关业务逻辑。

    参数：
        distance: distance 数据。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    return 1 / (1 + distance)


def calculate_group_damage_probability(target_data, own_position):
    """计算指定指标或中间结果，处理groupdamageprobability相关数据。

    参数：
        target_data: 目标数据。
        own_position: own位置坐标。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    total_damage_probability = 0
    total_firepower = 0

    for target in target_data:
        firepower = target[1]
        drone_count = target[2]
        defense_level = target[3]

        target_position = target[4:6]
        distance = np.sqrt((target_position[0] - own_position[0]) ** 2 +
                           (target_position[1] - own_position[1]) ** 2)

        decay_factor = distance_decay_factor(distance)

        damage_probability = (firepower * drone_count) / (defense_level * decay_factor)
        total_damage_probability += damage_probability
        total_firepower += firepower * drone_count

    if total_firepower > 0:
        group_damage_probability = total_damage_probability / total_firepower
    else:
        group_damage_probability = 0

    return group_damage_probability

own_position = (0, 0)

group_damage_probability = calculate_group_damage_probability(target_data, own_position)

print(f'目标群的总体毁伤概率: {group_damage_probability:.2f}')

def analyze_resource_redundancy(group_damage_probability, required_probability, redundancy_threshold=0.2):
    """处理analyze资源redundancy相关业务逻辑。

    参数：
        group_damage_probability: groupdamageprobability。
        required_probability: requiredprobability。
        redundancy_threshold: redundancythreshold。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    if group_damage_probability > required_probability * (1 + redundancy_threshold):
        print(f'目标群的资源存在冗余，当前毁伤概率: {group_damage_probability:.2f}, 需求毁伤概率: {required_probability:.2f}')

required_damage_probability = 0.7

analyze_resource_redundancy(group_damage_probability, required_damage_probability)
