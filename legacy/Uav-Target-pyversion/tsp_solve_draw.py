import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import distance_matrix
import Target_Screen
import plot_figure
import Kmeans_step1
import PSO_classify

matplotlib.rcParams['font.family'] = 'SimHei'  
matplotlib.rcParams['axes.unicode_minus'] = False

base = np.array([0, -16000])

target_data = np.array([
    [3, 156, 6, 2],  
    [6, 885, -386, 1],
    [17, 1167, -262, 3],
    [30, 1135, -282, 2],
    [34, 445, -217, 2],
    [36, 812, 93, 1],
    [39, 365, 176, 3],
    [67, 543, -493, 5],  
    [68, 351, -524, 2],
    [106, 330, -62, 2],
    [136, 219, -86, 4],
    [138, 275, -308, 1]
])


indices = target_data[:, 0].astype(int)
targets = target_data[:, 1:3] * 10
types = target_data[:, 3].astype(int)


all_points = np.vstack([base, targets])


dist_matrix = distance_matrix(all_points, all_points)


def tsp_greedy(dist_matrix):
    n = len(dist_matrix)
    visited = [0]  
    total_distance = 0
    current_point = 0

    while len(visited) < n:
        distances = dist_matrix[current_point]
        nearest_point = np.argmin([distances[j] if j not in visited else np.inf for j in range(n)])

        total_distance += dist_matrix[current_point, nearest_point]
        visited.append(nearest_point)
        current_point = nearest_point

    
    total_distance += dist_matrix[current_point, 0]
    visited.append(0)
    return visited, total_distance


optimal_order, total_distance = tsp_greedy(dist_matrix)


target_sequence = [indices[i - 1] for i in optimal_order[1:-1]]
print("打击目标顺序：", target_sequence)
print("决策路径总距离：", total_distance)


plt.figure(figsize=(10, 8))


marker_dict = {
    1: ('o', 'r'),  
    2: ('s', 'g'),  
    3: ('^', 'b'),  
    4: ('p', 'm'),  
    5: ('D', 'y')  
}
for i in range(1, len(optimal_order) - 1):
    idx = optimal_order[i]
    target_type = types[idx - 1]
    marker, color = marker_dict.get(target_type, ('o', 'k'))
    plt.scatter(all_points[idx, 0], all_points[idx, 1], color=color, marker=marker, s=100)
    plt.text(all_points[idx, 0] + 10, all_points[idx, 1] + 10, f'{indices[idx - 1]}', color='black', fontsize=12)


plt.scatter(base[0], base[1], color='red', s=100, label='UAV Base')
plt.text(base[0] + 20, base[1] - 30, 'UAV Base', color='red', fontsize=12)


for i in range(0, len(optimal_order) - 1):  
    start, end = all_points[optimal_order[i]], all_points[optimal_order[i + 1]]
    plt.plot([start[0], end[0]], [start[1], end[1]], 'k--')

    
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    plt.arrow(mid_x, mid_y, dx * 0.1, dy * 0.1, head_width=20, head_length=30, fc='gray', ec='gray')


legend_labels = {
    1: "地面装甲",
    2: "防御阵地",
    3: "雷达监测站",
    4: "通信枢纽",
    5: "指挥部"
}
for target_type in marker_dict:
    marker, color = marker_dict[target_type]
    plt.scatter([], [], color=color, marker=marker, s=100, label=f'{legend_labels[target_type]}')


plt.xlabel('X 轴')
plt.ylabel('Y 轴')
plt.title('打击路径决策')
plt.legend()
plt.grid()
plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\打击路径及次序.svg", dpi=600, format="svg")
plt.show()

data_Target = np.loadtxt(open('test_data/battlefield_target_data.csv'), delimiter=",", skiprows=1)
data_UAV = np.loadtxt(open('test_data/UAV.csv'), delimiter=",", skiprows=1)


data2 = Target_Screen.Target_Screen(data_Target)

plot_figure.plot_targets(data_Target)
plot_figure.plot_uav_counts(data_UAV)

print("战场态势初始化完成！")

cluster_1 = Kmeans_step1.Kmeans_step1(data_Target, data2, 2)


destroy_target_set = []
cluster_1_indices = [int(arr[0]) for arr in cluster_1[1]]
for index in cluster_1_indices:
    for row in data_Target:
        if int(row[0]) == index:
            target_row = row
            for element in cluster_1[1]:
                if int(element[0]) == index:
                    cluster_row = element
                    combined_data = list(target_row) + list(cluster_row)
                    destroy_target_set.append(combined_data)

plot_figure.plot_Destroy_targets(np.array(destroy_target_set))


destroy_target_array = np.array(destroy_target_set)
extracted_data = destroy_target_array[:, :7]
num_clusters = 8
max_iterations = 10
result_clusters, clustered_data, cluster_centers = PSO_classify.target_clustering_visualization(extracted_data,
                                                                                                num_clusters,
                                                                                                max_iterations)

