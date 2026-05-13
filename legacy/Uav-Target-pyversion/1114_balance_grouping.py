"""历史版本中的1114balancegrouping脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import matplotlib.pyplot as plt

data_Target = np.loadtxt(open('data_Target_extracted.csv'), delimiter=",", skiprows=1)
print(data_Target)

num_clusters = 8
max_iterations = 10
convergence_threshold = 2

type_2_indices = np.where(data_Target[:, 3] == 2)[0]
num_type_2 = len(type_2_indices)
initial_centers = data_Target[type_2_indices, 1:3][:min(num_clusters, num_type_2)]

ideal_cluster_size = len(data_Target) // num_clusters

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

    cluster_sizes = np.array([len(clusters[i]) for i in range(num_clusters)])
    size_diff = cluster_sizes - ideal_cluster_size
    size_penalty = np.abs(size_diff)

    if np.all(size_penalty < convergence_threshold):
        print(f"Convergence reached at iteration {iteration + 1}")
        break

    compactness = 0
    for i in range(num_clusters):
        if len(clusters[i]) > 1:
            positions = data_Target[clusters[i], 1:3]
            pairwise_distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
            compactness += np.sum(pairwise_distances) / len(clusters[i])
    compactness_penalty = compactness

    total_penalty = np.sum(size_penalty) + compactness_penalty
    print(f"Iteration {iteration + 1}: Total Penalty = {total_penalty:.2f}")

    while np.any(size_penalty > 0):
        for i in range(num_clusters):
            if size_diff[i] > 0:
                cluster_points = np.array(clusters[i])
                farthest_point = None
                max_distance = -np.inf
                for point in cluster_points:
                    distances = np.linalg.norm(data_Target[point, 1:3] - initial_centers, axis=1)
                    assigned_cluster = np.argmin(distances)
                    if distances[assigned_cluster] > max_distance:
                        max_distance = distances[assigned_cluster]
                        farthest_point = point
                clusters[i].remove(farthest_point)
                distances = np.linalg.norm(data_Target[farthest_point, 1:3] - initial_centers, axis=1)
                closest_cluster = np.argmin(distances)
                clusters[closest_cluster].append(farthest_point)

        new_centers = np.zeros((num_clusters, 2))
        for i in range(num_clusters):
            if clusters[i]:
                new_centers[i] = data_Target[clusters[i], 1:3].mean(axis=0)
            else:
                new_centers[i] = initial_centers[min(i, len(initial_centers) - 1)]

        initial_centers = new_centers

        cluster_sizes = np.array([len(clusters[i]) for i in range(num_clusters)])
        size_diff = cluster_sizes - ideal_cluster_size
        size_penalty = np.abs(size_diff)

plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    cluster_points = data_Target[clusters[i]]
    plt.scatter(cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i + 1}', s=50)
    for index in clusters[i]:
        plt.annotate(int(index),
                     (cluster_points[clusters[i].index(index), 1], cluster_points[clusters[i].index(index), 2]),
                     fontsize=8, ha='center', va='bottom')
    if i < len(initial_centers):
        plt.scatter(initial_centers[i, 0], initial_centers[i, 1], marker='x', color='black', s=100)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Target Clustering Visualization')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid()
plt.show()

for i in range(num_clusters):
    print(f'群 {i + 1}: 目标数量 = {len(clusters[i])}, 索引 = {clusters[i]}')
