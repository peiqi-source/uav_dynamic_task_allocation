import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data_Target_extracted = np.array([
    [1, -388, 1090, 1, 3, 2],
    [2, -479, 999, 1, 4, 2],
    [3, 156, 6, 1, 3, 2],
    [4, 724, 665, 2, 3, 2],
    [5, 982, -594, 1, 4, 2],
    [6, 885, -386, 1, 3, 2],
    [7, 1138, -1322, 2, 1, 2],
    [8, -1871, 1048, 1, 2, 2],
    [9, -943, 735, 2, 4, 2],
    [10, -405, -1017, 2, 2, 2],
    [11, 740, -1411, 2, 4, 2],
    [12, 1453, 1077, 2, 4, 2],
    [13, -1413, 93, 2, 2, 2],
    [14, -1942, 149, 2, 3, 2],
    [15, 491, -1114, 2, 4, 2],
    [16, -1794, -1253, 2, 1, 2],
    [17, 1167, -262, 2, 2, 2],
    [18, -1575, -1288, 2, 3, 2],
    [19, -647, 239, 2, 2, 2],
    [20, 330, 1120, 2, 2, 2],
    [21, -687, 658, 2, 1, 2],
    [22, 523, 1349, 2, 4, 2],
    [23, -330, -108, 2, 1, 2],
    [24, 297, -853, 2, 3, 2],
    [25, -1945, -1188, 2, 4, 2],
    [26, -1743, -712, 2, 3, 2],
    [27, -1957, -822, 2, 2, 2],
    [28, 1959, -707, 2, 1, 2],
    [29, 1655, -992, 2, 1, 2],
    [30, 1135, -282, 2, 4, 2],
    [31, -1368, -1439, 2, 4, 2],
    [32, -1494, -625, 2, 4, 2],
    [33, 37, -400, 2, 1, 2],
    [34, 445, -217, 2, 1, 2],
    [35, 483, -925, 2, 3, 2],
    [36, 812, 93, 2, 4, 2],
    [37, 9, 557, 2, 1, 2],
    [38, -657, 74, 2, 4, 2],
    [39, 365, 176, 2, 1, 2],
    [40, 73, -189, 2, 1, 2],
    [43, 18, -1191, 1, 3, 1],
    [57, 132, -735, 1, 4, 1],
    [67, 543, -493, 1, 3, 1],
    [68, 351, -524, 1, 3, 1],
    [89, -390, -130, 1, 2, 1],
    [104, -308, -481, 1, 3, 1],
    [106, 330, -62, 1, 4, 1],
    [119, 167, -1169, 1, 2, 1],
    [123, -689, -443, 1, 1, 1],
    [124, -251, -209, 1, 1, 1],
    [128, -361, -501, 1, 1, 1],
    [136, 219, -86, 1, 1, 1],
    [138, 275, -308, 1, 1, 1],
    [141, -342, -1028, 1, 4, 1],
    [147, 60, -70, 1, 4, 1],
    [149, 402, -697, 1, 2, 1],
    [151, 370, -772, 1, 3, 1],
    [158, -631, -225, 1, 2, 1],
    [167, 199, -1097, 1, 3, 1],
    [171, -671, -381, 1, 1, 1],
    [174, 4, -9, 1, 3, 1],
    [196, 86, -264, 1, 3, 1]
])

coordinates = data_Target_extracted[:, 1:3]

num_clusters = 8
kmedoids = KMedoids(n_clusters=num_clusters, random_state=0).fit(coordinates)
labels = kmedoids.labels_

cluster_counts = np.bincount(labels)

def balance_clusters(data, labels, cluster_counts, target_count):
    
    medoids = kmedoids.cluster_centers_

    for cluster_idx in range(num_clusters):
        while cluster_counts[cluster_idx] > target_count:
            
            cluster_points = data[labels == cluster_idx, 1:3]
            distances = cdist(cluster_points, [medoids[cluster_idx]])
            farthest_idx = np.argmax(distances)
            farthest_point = cluster_points[farthest_idx]

            other_clusters = np.setdiff1d(range(num_clusters), [cluster_idx])
            other_distances = cdist([farthest_point], medoids[other_clusters])
            nearest_cluster = other_clusters[np.argmin(other_distances)]

            point_idx = np.where((data[:, 1:3] == farthest_point).all(axis=1))[0][0]
            labels[point_idx] = nearest_cluster

            cluster_counts[cluster_idx] -= 1
            cluster_counts[nearest_cluster] += 1

    return labels
target_count = len(data_Target_extracted) // num_clusters
balanced_labels = balance_clusters(data_Target_extracted, labels, cluster_counts, target_count)

for i in range(num_clusters):
    print(f"簇 {i + 1}: {np.sum(balanced_labels == i)} 个目标点")
colors = cm.rainbow(np.linspace(0, 1, num_clusters))
plt.figure(figsize=(10, 8))
for cluster_idx in range(num_clusters):
    cluster_points = data_Target_extracted[balanced_labels == cluster_idx]
    plt.scatter(cluster_points[:, 1], cluster_points[:, 2], s=100, color=colors[cluster_idx], label=f'Cluster {cluster_idx + 1}')
medoids = kmedoids.cluster_centers_
plt.xlabel('X 坐标')
plt.ylabel('Y 坐标')
plt.title('平衡调整后的 K-Medoids 聚类结果')
plt.legend()
plt.grid()
plt.show()
