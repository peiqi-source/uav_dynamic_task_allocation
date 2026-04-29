import numpy as np

target_data = np.array([
    [1, 100, 5, 50, 10, 20],
    [2, 150, 3, 70, 15, 25],
    [3, 120, 10, 40, 20, 30],
    [4, 90, 8, 60, 25, 35]
], dtype=float)

def distance_decay_factor(distance):
    return 1 / (1 + distance)


def calculate_group_damage_probability(target_data, own_position):
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
    if group_damage_probability > required_probability * (1 + redundancy_threshold):
        print(f'目标群的资源存在冗余，当前毁伤概率: {group_damage_probability:.2f}, 需求毁伤概率: {required_probability:.2f}')

required_damage_probability = 0.7

analyze_resource_redundancy(group_damage_probability, required_damage_probability)
