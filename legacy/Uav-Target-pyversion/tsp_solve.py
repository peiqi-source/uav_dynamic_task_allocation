"""历史版本中的tspsolve脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import distance_matrix


def plot_strike_path(target_data, base):

    """读取指标数据并生成可视化图表，处理打击路径相关数据。

    参数：
        target_data: 目标数据。
        base: 基础。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = False


    indices = target_data[:, 0].astype(int)
    targets = target_data[:, 1:3]
    targets *= 10
    types = target_data[:, 6].astype(int)

    all_points = np.vstack([base, targets])

    dist_matrix = distance_matrix(all_points, all_points)

    def tsp_greedy(dist_matrix):
        """处理tspgreedy相关业务逻辑。

        参数：
            dist_matrix: distmatrix。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
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


    for i in range(1, len(optimal_order) - 2):
        start, end = all_points[optimal_order[i]], all_points[optimal_order[i + 1]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'k--')


        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        plt.arrow(mid_x, mid_y, dx * 0.1, dy * 0.1, head_width=20 * 6, head_length=30 * 6, fc='gray', ec='gray')


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


    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\5_3_2\\打击路径决策.svg", dpi=600, format="svg")
    plt.show()

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import distance_matrix


matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False


base = np.array([0, -1600])


target_data = np.array([
    [3, 156, 6, 3],
    [6, 885, -386, 4],
    [17, 1167, -262, 5],
    [30, 1135, -282, 4],
    [34, 445, -217, 3],
    [36, 812, 93, 2],
    [39, 365, 176, 2],
    [67, 543, -493, 1],
    [68, 351, -524, 1],
    [106, 330, -62, 3],
    [136, 219, -86, 4],
    [138, 275, -308, 1]
])


indices = target_data[:, 0].astype(int)
targets = target_data[:, 1:3]
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


for i in range(1, len(optimal_order) - 2):
    start, end = all_points[optimal_order[i]], all_points[optimal_order[i + 1]]
    plt.plot([start[0], end[0]], [start[1], end[1]], 'k--')


    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    plt.arrow(mid_x, mid_y, dx * 0.1, dy * 0.1, head_width=20, head_length=30, fc='green', ec='green')


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
plt.show()
"""
