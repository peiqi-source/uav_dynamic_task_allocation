"""历史版本中的PSO 算法classify脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def target_clustering_visualization(data_Target, num_clusters, max_iterations):


    """处理目标clusteringvisualization相关业务逻辑。

    参数：
        data_Target: 数据目标。
        num_clusters: num目标簇集合。
        max_iterations: 最大值iterations。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False

    fitness_values = []

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


        fitness = 0
        for i in range(num_clusters):
            if clusters[i]:
                for index in clusters[i]:
                    fitness += np.min(np.linalg.norm(data_Target[index, 1:3] - initial_centers, axis=1)) ** 2
        fitness_values.append(fitness)


        new_centers = np.zeros((num_clusters, 2))
        for i in range(num_clusters):
            if clusters[i]:
                new_centers[i] = data_Target[clusters[i], 1:3].mean(axis=0)
            else:
                new_centers[i] = initial_centers[min(i, len(initial_centers) - 1)]

        initial_centers = new_centers

    clustered_data = {i + 1: [] for i in range(num_clusters)}
    for cluster_num in range(num_clusters):
        target_indices = clusters[cluster_num]
        for index in target_indices:
            clustered_data[cluster_num + 1].append(data_Target[index])
        clustered_data[cluster_num + 1] = np.array(clustered_data[cluster_num + 1])

    cluster_centers = {}
    for i in range(num_clusters):
        if i < len(initial_centers):
            cluster_centers[i + 1] = initial_centers[i] * 10
        else:

            cluster_centers[i + 1] = np.array([0, 0])
    """

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
    """

    plt.figure(figsize=(12, 8))
    color_list = ['r', 'g', 'b', 'm', 'y', 'c', 'orange', 'purple']
    for i in range(num_clusters):
        cluster_points = data_Target[clusters[i]]
        color = color_list[i % len(color_list)]
        plt.scatter(cluster_points[:, 1]*10, cluster_points[:, 2]*10, label=f'Cluster {i + 1}', s=50, color=color)

        for index in range(len(cluster_points)):
            plt.annotate(int(cluster_points[index, 0]),
                         (cluster_points[index, 1]*10, cluster_points[index, 2]*10),
                         fontsize=8, ha='center', va='bottom')
        if i < len(initial_centers):
            plt.scatter(initial_centers[i, 0]*10, initial_centers[i, 1]*10, marker='x', color='black', s=100)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title('目标分群结果图')
    plt.xlabel('X 轴')
    plt.ylabel('Y 轴')

    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\PSO目标分群结果颜色对应uav.svg", dpi=600, format="svg")
    plt.show()

    for i in range(num_clusters):
        target_labels = [int(data_Target[target_index][0]) for target_index in clusters[i]]
        print(f'群 {i + 1}: 目标数量 = {len(clusters[i])}, 目标标号 = {target_labels}')

    """

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_values) + 1), fitness_values, marker='o', color='b')
    plt.xlabel('迭代轮数')
    plt.ylabel('适应度函数值')
    plt.grid()
    plt.legend()
    plt.show()


    given_clusters_1 = {
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
    for cluster_index in given_clusters_1:
        indices = given_clusters_1[cluster_index]
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


    given_clusters_2 = {
        1: [48, 49, 57, 59, 22, 44, 45, 50, 9, 53],
        2: [7, 12, 13, 8, 20, 18, 37, 1, 0],
        3: [32, 39, 54, 60, 61, 2, 51, 52, 33, 43, 46],
        4: [3, 11, 19, 21, 36, 35, 38],
        5: [4, 5, 6, 27, 28, 5, 29, 16, 10, 42],
        6: [23, 40, 41, 47, 55, 56, 58, 34, 14],
        7: [15, 17, 24, 25, 26, 30, 31]

    }
    plt.figure(figsize=(10, 6))
    for cluster_index in given_clusters_2:
        indices = given_clusters_2[cluster_index]
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


    given_clusters_3 = {
        1: [48, 49, 57, 59, 22, 44, 45, 50, 9, 53, 18, 37],
        2: [32, 39, 54, 60, 61, 2, 51, 52, 33, 43, 46, 42],
        3: [3, 11, 19, 21, 36, 38, 1, 0, 8, 20],
        4: [4, 5, 6, 27, 28, 5, 29, 16, 35, 10],
        5: [23, 40, 41, 47, 55, 56, 58, 34, 14],
        6: [15, 17, 24, 25, 26, 30, 31, 7, 12, 13]
    }
    plt.figure(figsize=(10, 6))
    for cluster_index in given_clusters_3:
        indices = given_clusters_3[cluster_index]
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
    """

    return clusters, clustered_data, cluster_centers


"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False



data_Target = np.loadtxt(open('data_Target_extracted.csv'), delimiter=",", skiprows=1)
print(data_Target)


num_clusters = 8
max_iterations = 15
fitness_values = []


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


    fitness = 0
    for i in range(num_clusters):
        if clusters[i]:
            for index in clusters[i]:
                fitness += np.min(np.linalg.norm(data_Target[index, 1:3] - initial_centers, axis=1)) ** 2
    fitness_values.append(fitness)


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
    print(f'群 {i+1}: 目标数量 = {len(clusters[i])}, 索引 = {clusters[i]}')


plt.figure(figsize=(10, 6))

plt.plot(range(1, len(fitness_values) + 1), fitness_values, marker='o', color='b')

plt.xlabel('迭代轮数')
plt.ylabel('适应度函数值')
plt.grid()
plt.legend()
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




given_clusters = {
    1: [48, 49, 57, 59, 22, 44, 45, 50, 9, 53],
    2: [7, 12, 13, 8, 20, 18, 37, 1, 0],
    3: [32, 39, 54, 60, 61, 2, 51, 52, 33, 43, 46],
    4: [3, 11, 19, 21, 36, 35, 38],
    5: [4, 5, 6, 27, 28, 5, 29, 16, 10,  42],
    6: [23, 40, 41, 47, 55, 56, 58, 34, 14],
    7: [15, 17, 24, 25, 26, 30, 31]

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



given_clusters = {
    1: [48, 49, 57, 59, 22, 44, 45, 50, 9, 53, 18, 37],
    2: [32, 39, 54, 60, 61, 2, 51, 52, 33, 43, 46, 42],
    3: [3, 11, 19, 21, 36,  38, 1, 0,  8, 20,],
    4: [4, 5, 6, 27, 28, 5, 29, 16, 35, 10],
    5: [23, 40, 41, 47, 55, 56, 58, 34, 14],
    6: [15, 17, 24, 25, 26, 30, 31, 7, 12, 13]
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
"""

