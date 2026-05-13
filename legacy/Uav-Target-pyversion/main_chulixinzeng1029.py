"""历史版本中的mainchulixinzeng1029脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import UAV
import matplotlib.pyplot as plt
import Target_Screen
import Kmeans_step1
import Kmeans_step2
import Particle_algorithm
import joblib
import random
import aco_plot
import pandas as pd

def calculate_angle(x, y):
    """计算指定指标或中间结果，处理angle 数据相关数据。

    参数：
        x: 横坐标。
        y: 纵坐标。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    angle = np.arctan2(y, x)
    angle_degrees = np.degrees(angle)
    if angle_degrees < 0:
        angle_degrees += 360
    return angle_degrees

def assess_combat_resources(data_Target, data_UAV):
    """处理assesscombat资源集合相关业务逻辑。

    参数：
        data_Target: 数据目标。
        data_UAV: 数据无人机。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    total_resources_required = sum(data_Target[:, -1])
    available_resources = sum(data_UAV[:, -1])
    if total_resources_required <= available_resources:
        return True
    else:
        return False

data_UAV = np.loadtxt(open('test_data/UAV.csv'), delimiter=",", skiprows=1)
data_Target = np.loadtxt(open('test_data/battlefield_target_data.csv'), delimiter=",", skiprows=1)

d, attack, cluster_results, test1 = UAV.UAV(data_UAV)

data2 = Target_Screen.Target_Screen(data_Target, attack)

cluster_1 = Kmeans_step1.Kmeans_step1(data_Target, data2, 2)
print(cluster_1)

cluster = Kmeans_step2.Kmeans_step2(data_Target, cluster_1, d)

target_indices = [int(item[0]) for item in cluster_1[1]]

extracted_rows = []

for index in target_indices:

    matching_rows = data_Target[data_Target[:, 0] == index]
    extracted_rows.append(matching_rows)

data_Target_extracted = np.vstack(extracted_rows) if extracted_rows else np.empty((0, data_Target.shape[1]))
df = pd.DataFrame(data_Target_extracted, columns=['num', 'x', 'y', 'type', 'defense', 'significance'])

df.to_csv('data_Target_extracted.csv', index=False)

print("data_Target_extracted = ", data_Target_extracted)

num_clusters = 8
max_iterations = 10

type_2_indices = np.where(data_Target_extracted[:, 3] == 2)[0]
num_type_2 = len(type_2_indices)
initial_centers = data_Target_extracted[type_2_indices, 1:3][:min(num_clusters, num_type_2)]

for iteration in range(max_iterations):
    clusters = {i: [] for i in range(num_clusters)}

    for i in range(len(data_Target_extracted)):
        if data_Target_extracted[i, 3] == 2:
            cluster_index = np.argmin(np.linalg.norm(initial_centers - data_Target_extracted[i, 1:3], axis=1))
            clusters[cluster_index].append(i)
        else:
            distances = np.linalg.norm(data_Target_extracted[i, 1:3] - initial_centers, axis=1)
            assigned_cluster = np.argmin(distances)
            clusters[assigned_cluster].append(i)

    for i in range(len(data_Target_extracted)):
        if not any(i in clusters[c] for c in clusters):
            distances = np.linalg.norm(data_Target_extracted[i, 1:3] - initial_centers, axis=1)
            assigned_cluster = np.argmin(distances)
            clusters[assigned_cluster].append(i)

    new_centers = np.zeros((num_clusters, 2))
    for i in range(num_clusters):
        if clusters[i]:
            new_centers[i] = data_Target_extracted[clusters[i], 1:3].mean(axis=0)
        else:
            new_centers[i] = initial_centers[min(i, len(initial_centers) - 1)]

    if np.all(np.isclose(new_centers, initial_centers)):
        break
    initial_centers = new_centers

cluster_data_extracted = {}

for i in range(num_clusters):
    cluster_data_extracted[f'cluster_{i+1}_data'] = data_Target_extracted[clusters[i]]
    print(f'Cluster {i+1} Data:')
    print(cluster_data_extracted[f'cluster_{i+1}_data'])
    print('-' * 50)

plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    cluster_points = data_Target_extracted[clusters[i]]
    plt.scatter(cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i+1}', s=50)
    if i < len(initial_centers):
        plt.scatter(initial_centers[i, 0], initial_centers[i, 1], marker='x', color='black', s=100)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Target Clustering Visualization')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid()
plt.show()

for i in range(num_clusters):
    print(f'群 {i+1}: 目标数量 = {len(clusters[i])}, 索引 = {clusters[i]}')

for i in range(num_clusters):
    print(f'最终群 {i+1}: 目标数量 = {len(clusters[i])}, 索引 = {clusters[i]}')

target_data = cluster_1[1]
values = np.array(target_data)[:, 3]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = plt.get_cmap('tab20', num_clusters)

for i in range(num_clusters):
    cluster_points = data_Target_extracted[clusters[i]]
    ax.scatter(cluster_points[:, 1], cluster_points[:, 2], values[clusters[i]],
               label=f'Cluster {i + 1}', s=100, color=colors(i), alpha=0.7)

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Value (v)')
ax.set_zticks(np.arange(0, 1.1, 0.2))
ax.set_title('3D Target Clustering Visualization')
ax.legend()

plt.show()
"""

num_new_targets = random.randint(1, 3)
new_targets = []

for _ in range(num_new_targets):

    new_target_index = len(data_Target) + 1
    new_target_x = random.randint(-1000, 1000)
    new_target_y = random.randint(0, 2000)
    new_target_type = random.randint(1, 2)
    new_target_defense = random.randint(1, 4)
    new_target_importance = random.randint(1, 2)

    new_target_info = [new_target_index, new_target_x, new_target_y, new_target_type, new_target_defense,
                   new_target_importance]
    new_targets.append(new_target_info)
    data_Target = np.vstack([data_Target, new_target_info])

data2 = Target_Screen.Target_Screen(data_Target, attack)


rf_clf = joblib.load('random_forest_model.joblib')
new_features = data2[-num_new_targets:, 3:6]
predictions = rf_clf.predict(new_features)

destroy_targets = []

for i, (prediction, new_target_info) in enumerate(zip(predictions, new_targets),
                                                  start=len(data_Target) - num_new_targets + 1):
    print(f"New target {new_target_info[0]}: {'Destroy' if prediction == 1 else 'Do not destroy'}")
    if prediction == 1:
        new_target_index = new_target_info[0]
        target_data = data2[data2[:, 0] == new_target_index][:, :6]
        if target_data.size > 0:
            target_data = target_data[0]
            target_data = np.append(target_data, new_target_info[4])
            destroy_targets.append(target_data)

if destroy_targets:
    destroy_targets = np.array(destroy_targets)
    cluster_1[1] = np.vstack([cluster_1[1], destroy_targets])


    if not assess_combat_resources(cluster_1[1], data_UAV):
        print(f"进攻资源有限，筛选摧毁目标集!")

        num_to_remove = np.sum(predictions == 1)
        sorted_indices = np.argsort(cluster_1[1][:, 3])
        removed_indices = sorted_indices[:num_to_remove]
        removed_targets = cluster_1[1][removed_indices, 0]
        cluster_1[1] = np.delete(cluster_1[1], removed_indices, axis=0)

        for removed_index in removed_targets:
            print(f"受进攻资源限制，目标 {int(removed_index)} 从摧毁目标集清除!")

    targets = cluster_1[1]

    plt.figure(1)
    plt.title('摧毁目标集')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim(-50, 50)
    plt.ylim(-1, 100)
    plt.plot(0, 0, 'bp', markerfacecolor='r', markersize=15)
    plt.text(0 + 0.01, 0, 'Base')

    for target in cluster_1[1]:
        original_position = data_Target[data_Target[:, 0] == target[0], 1:3]
        original_type = data_Target[data_Target[:, 0] == target[0], 3]
        x, y = original_position[0]
        plt.plot(x, y, 'o', color=[0.5, 0.5, 0.5], markerfacecolor='g')
        plt.text(x + 0.01, y, str(target[0]), visible=True)
    plt.grid(True)
    plt.show()

    for target in cluster_1[1]:
        original_position = data_Target[data_Target[:, 0] == target[0], 1:3]
        x, y = original_position[0]
        angle = calculate_angle(x, y)
        target[-2] = angle

    num_clusters = 3
    best_solution, best_value, best_assignment = Particle_algorithm.pso_optimization(targets, num_clusters)


    print("最佳分配方案:", best_assignment)

    colors = ['red', 'green', 'blue']

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(0, 0, 0, c='r', marker='*', label='飞行器基地')
    ax.text(0 + 0.01, 0, 0, '飞行器基地')
    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, x in enumerate(best_assignment) if x == cluster_id]
        cluster_targets = targets[cluster_indices]

        original_positions = np.array(
            [data_Target[data_Target[:, 0] == target[0], 1:4] for target in cluster_targets]).reshape(-1, 3)
        x = original_positions[:, 0]
        y = original_positions[:, 1]
        z = cluster_targets[:, 3]
        indices = cluster_targets[:, 0]

        ax.scatter(x, y, z, color=colors[cluster_id], label=f'Cluster {cluster_id + 1}')

        for j in range(len(x)):
            ax.text(x[j], y[j], z[j], f'{int(indices[j])}', color='black')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Value')
    ax.set_title('摧毁目标集分类')
    ax.legend()
    plt.show()

    updated_clusters = []
    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, x in enumerate(best_assignment) if x == cluster_id]
        cluster_targets = targets[cluster_indices]

        updated_cluster = np.array(
            [[target[0], data_Target[data_Target[:, 0] == target[0], 1][0],
              data_Target[data_Target[:, 0] == target[0], 2][0],
              data2[data2[:, 0] == target[0], -1][0]] for target in cluster_targets]
        )
        updated_cluster = updated_cluster[updated_cluster[:, -1].argsort()[::-1]]
        updated_clusters.append(updated_cluster)

    cluster = updated_clusters

    aco_plot.plot_clusters_and_routes(cluster)

else:
    print(f"New target {new_target_index} not added to destroy targets set.")


    colors = ['red', 'green', 'blue', 'black']
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(-45, 35)
    ax.scatter(0, 0, 0, c='r', marker='*', label='飞行器基地')
    ax.text(0 + 0.01, 0, 0, '飞行器基地')
    for i, cluster_group in enumerate(cluster):
        if i >= len(colors):
            color = colors[i % len(colors)]
        else:
            color = colors[i]

        x = cluster_group[:, 1]
        y = cluster_group[:, 2]
        z = cluster_group[:, 3]
        indices = cluster_group[:, 0]

        ax.scatter(x, y, z, c=color, label=f'Cluster {i + 1}')
        for j in range(len(x)):
            ax.text(x[j], y[j], z[j], f'{int(indices[j])}', color='black')

    ax.set_title('摧毁目标集分类')
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('角度')
    ax.legend(loc='upper right')
    ax.set_xticks(np.arange(-50, 51, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_zticks(np.arange(0, 1.1, 0.2))
    ax.grid(True)
    plt.show()

    aco_plot.plot_clusters_and_routes(cluster)"""

