import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, distance_matrix
from sklearn.cluster import KMeans

# 数据
"""data_Target = np.array([
    [1, -45, 28, 2, 4, 2],
    [2, 1, 74, 2, 4, 2],
    [3, 35, 42, 2, 3, 2],
    [4, 24, 46, 1, 2, 1],
    [5, -18, 38, 1, 2, 1],
    [6, 38, 21, 1, 2, 1],
    [7, -17, 66, 1, 2, 1],
    [8, -49, 44, 1, 2, 1],
    [9, 29, 7, 1, 2, 1],
    [10, 3, 96, 1, 1, 1],
    [11, -12, 15, 1, 1, 1],
    [12, 43, 45, 1, 1, 1],
    [13, -7, 88, 1, 2, 1],
    [14, -56, 40, 1, 1, 1],
    [15, 15, 48, 2, 4, 2],
    [16, -25, 33, 2, 4, 2],
    [17, -11, 29, 1, 1, 1],
    [18, -31, 22, 1, 1, 1],
    [19, 54, 29, 2, 4, 2],
    [20, -36, 61, 2, 3, 2]
])"""


data_Target = np.loadtxt(open('data_Target_extracted.csv'), delimiter=",", skiprows=1)
# print(data_Target)

# PSO参数
num_clusters = 8
max_iterations = 10

total_targets = len(data_Target)
avg_targets_per_cluster = total_targets // num_clusters

type_2_indices = np.where(data_Target[:, 3] == 2)[0]
num_type_2 = len(type_2_indices)
initial_centers = data_Target[type_2_indices, 1:3][:min(num_clusters, num_type_2)]

for iteration in range(max_iterations):
    clusters = {i: [] for i in range(num_clusters)}

    for i in range(len(data_Target)):
        if data_Target[i, 3] == 2:
            cluster_index = np.argmin(np.linalg.norm(initial_centers - data_Target[i, 1:3], axis=1))
            clusters[cluster_index].append(i)
        else:
            distances = np.linalg.norm(data_Target[i, 1:3] - initial_centers, axis=1)
            assigned_cluster = np.argmin(distances)
            clusters[assigned_cluster].append(i)

    for i in range(len(data_Target)):
        if not any(i in clusters[c] for c in clusters):
            distances = np.linalg.norm(data_Target[i, 1:3] - initial_centers, axis=1)
            assigned_cluster = np.argmin(distances)
            clusters[assigned_cluster].append(i)

    new_centers = np.zeros((num_clusters, 2))
    for i in range(num_clusters):
        if clusters[i]:
            new_centers[i] = data_Target[clusters[i], 1:3].mean(axis=0)
        else:
            new_centers[i] = initial_centers[min(i, len(initial_centers) - 1)]

    if np.all(np.isclose(new_centers, initial_centers)):
        break
    initial_centers = new_centers

plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    cluster_points = data_Target[clusters[i]]
    plt.scatter(cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i+1}', s=50)
    for j, index in enumerate(clusters[i]):
        true_index = int(data_Target[index, 0])
        plt.annotate(true_index, (cluster_points[j, 1], cluster_points[j, 2]), fontsize=8, ha='center', va='bottom')
    if i < len(initial_centers):
        plt.scatter(initial_centers[i, 0], initial_centers[i, 1], marker='x', color='black', s=100)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Target Clustering Visualization')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid()
plt.show()

for i in range(num_clusters):
    true_indices = [int(data_Target[index, 0]) for index in clusters[i]]
    print(f'群 {i + 1}: 目标数量 = {len(clusters[i])}, 索引 = {true_indices}')

for i in range(num_clusters):
    true_indices = [int(data_Target[index, 0]) for index in clusters[i]]
    cluster_points = data_Target[clusters[i], 1:3]
    if len(cluster_points) > 1:
        std_x, std_y = cluster_points[:, 0].std(), cluster_points[:, 1].std()
        std_dev = (std_x, std_y)

        dist_matrix = distance_matrix(cluster_points, cluster_points)
        np.fill_diagonal(dist_matrix, np.inf)
        avg_nearest_neighbor_dist = dist_matrix.min(axis=1).mean()

        try:
            hull = ConvexHull(cluster_points)
            convex_hull_area = hull.volume
        except Exception as e:
            convex_hull_area = 0.0
    else:
        std_dev = (0, 0)
        avg_nearest_neighbor_dist = 0
        convex_hull_area = 0

    print(f'群 {i + 1}:')
    print(f'  目标数量 = {len(clusters[i])}, 索引 = {true_indices}')
    print(f'  坐标标准差 = {std_dev}')
    print(f'  平均最近邻距离 = {avg_nearest_neighbor_dist:.2f}')
    print(f'  凸包面积 = {convex_hull_area:.2f}')

    if avg_nearest_neighbor_dist < 190 and len(clusters[i]) > avg_targets_per_cluster:
        cluster_points = data_Target[clusters[i], 1:3]
        distances_to_center = np.linalg.norm(cluster_points - new_centers[i], axis=1)
        sorted_indices = np.argsort(distances_to_center)[::-1]

        num_to_remove = len(clusters[i]) - avg_targets_per_cluster

        keep_indices = sorted_indices[num_to_remove - 2:]
        true_indices = [true_indices[idx] for idx in keep_indices]
        clusters[i] = [clusters[i][idx] for idx in keep_indices]

plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    cluster_points = data_Target[clusters[i]]
    plt.scatter(cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i + 1}', s=50)
    for index in clusters[i]:
        true_index = int(data_Target[index, 0])
        plt.annotate(true_index,
                     (cluster_points[clusters[i].index(index), 1], cluster_points[clusters[i].index(index), 2]),
                     fontsize=8, ha='center', va='bottom')
    if i < len(initial_centers):
        plt.scatter(initial_centers[i, 0], initial_centers[i, 1], marker='x', color='black', s=100)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Processed Target Clustering Visualization')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid()
plt.show()

given_clusters = {
    1: [48, 49, 57, 59, 22, 44, 45, 50, 9, 53],
    2: [7, 12, 13, 8, 20, 18, 37],
    3: [32, 39, 54, 60, 61, 2, 51, 52],
    4: [3, 11, 19, 21, 1, 0, 36],
    5: [4, 6, 27, 28, 5, 29, 16, 10],
    6: [5, 33, 35, 38, 42, 43, 46],
    7: [14, 23, 34, 40, 41, 47, 55, 56, 58],
    8: [15, 17, 24, 25, 26, 30, 31]
}
plt.figure(figsize=(10, 6))
for cluster_index in given_clusters:
    indices = given_clusters[cluster_index]
    cluster_points = data_Target[indices]
    plt.scatter(cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster_index}', s=50)
    for index in indices:
        plt.annotate(int(index), (cluster_points[indices.index(index), 1], cluster_points[indices.index(index), 2]),
                     fontsize=8, ha='center', va='bottom')

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Second Target Clustering Visualization')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid()
plt.show()

