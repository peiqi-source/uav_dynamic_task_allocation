"""历史版本中的help 数据脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np

np.random.seed()
num_targets = 20
target_indices = np.arange(201, 201 + num_targets)
positions_x = np.random.uniform(-2000, 1900, num_targets)
positions_y = np.random.uniform(-2000, 1800, num_targets)
types = np.random.choice([1, 2], num_targets)
defenses = np.random.randint(1, 5, num_targets)
importances = np.random.randint(1, 3, num_targets)

destroyed = np.zeros(num_targets, dtype=int)
for i in range(num_targets):
    if types[i] == 2:

        destroyed[i] = np.random.choice([0, 1], p=[0.2, 0.8])
    else:
        destroyed[i] = np.random.choice([0, 1])

target_data = np.column_stack((target_indices, positions_x, positions_y, types, defenses, importances, destroyed))
destroyed_targets = target_data[target_data[:, -1] == 1]
sorted_destroyed_targets = destroyed_targets[destroyed_targets[:, 0].argsort()]

print("目标数据：")
print("索引, 位置X, 位置Y, 类型, 防御力, 重要性, 是否摧毁")
for target in sorted_destroyed_targets:
    print(target)
