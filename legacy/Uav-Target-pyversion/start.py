"""
import numpy as np

# 定义环境和动作空间
num_targets = 20
num_agents = 12
num_actions = num_targets

# 初始化Q表
Q = np.zeros((num_agents, num_targets, num_actions))

# 设置超参数
epsilon = 0.1
alpha = 0.1
gamma = 0.9
num_episodes = 1000


# 定义动作选择函数
def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(num_actions)
    else:
        return np.argmax(Q[state[0], state[1], :])


# 主训练循环
for episode in range(num_episodes):
    state = (0, 0)  # 初始状态
    total_reward = 0

    for agent in range(num_agents):
        action = choose_action(state, epsilon)

        # 模拟执行动作，得到奖励(实际应用中需要根据具体情况计算奖励)
        reward = np.random.uniform(0, 1)

        # 更新Q值
        next_state = (agent, action)
        Q[state[0], state[1], action] += alpha * (
                    reward + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action])

        state = next_state
        total_reward += reward

    print("Episode: {}, Total Reward: {}".format(episode, total_reward))
"""

'''
cluster_1 = Kmeans_step1(data_Target, data2, 2)

cluster = Kmeans_step2(data_Target, cluster_1, d)

City = []
City1 = []
bestroute = []
bestroute1 = []

for i in range(d):
    A = cluster[i][1][:, [1, 2, 0]]
    B = cluster[i][1][:, 0]
    A1 = np.vstack(([0, 0, 0], A))
    B1 = np.hstack(([0], B))
    City.extend(A)
    City1.extend(A1)
    bestroute.extend(B)
    bestroute1.extend(B1)

City1.append([0, 0, 0])
bestroute1.append(0)

ACO_DrawPath(bestroute, City, bestroute1, City1)
'''

'''
import numpy as np
import random


def initialize_centroids(data, k):
    """K-means++ 初始化中心点"""
    n = data.shape[0]
    centroids = []
    centroids.append(data[np.random.randint(n)])

    for _ in range(1, k):
        distances = np.array([min(np.linalg.norm(x - centroid) ** 2 for centroid in centroids) for x in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = random.random()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(data[j])
                break

    return np.array(centroids)


def assign_clusters(data, centroids):
    """分配每个数据点到最近的中心点"""
    clusters = {}
    for x in data:
        best_centroid = \
        min([(i, np.linalg.norm(x - centroid)) for i, centroid in enumerate(centroids)], key=lambda t: t[1])[0]
        try:
            clusters[best_centroid].append(x)
        except KeyError:
            clusters[best_centroid] = [x]
    return clusters


def update_centroids(clusters):
    """更新中心点"""
    new_centroids = []
    for key in clusters.keys():
        new_centroids.append(np.mean(clusters[key], axis=0))
    return new_centroids


def calculate_sse(clusters, centroids):
    """计算 SSE（Sum of Squared Errors）"""
    sse = 0
    for key, cluster in clusters.items():
        centroid = centroids[key]
        sse += sum(np.linalg.norm(x - centroid) ** 2 for x in cluster)
    return sse


def kmeans(data, k, max_iterations=100, tol=1e-4):
    """智能 K-means 聚类算法"""
    centroids = initialize_centroids(data, k)
    previous_sse = float('inf')

    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        centroids = update_centroids(clusters)
        sse = calculate_sse(clusters, centroids)

        if abs(previous_sse - sse) < tol:
            break
        previous_sse = sse

    return centroids, clusters, sse


# 示例数据
data = np.array([[1, 2], [1, 4], [1, 0],
                 [4, 2], [4, 4], [4, 0]])

k = 2
centroids, clusters, sse = kmeans(data, k)

print(f"Centroids: {centroids}")
for key, cluster in clusters.items():
    print(f"Cluster {key}: {cluster}")
    
'''
'''
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def kmeans(X, K, max_iters=100, tol=1e-4):
    # 随机选择 K 个数据点作为初始簇中心
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]

    for iteration in range(max_iters):
        # 创建一个数组存储每个点的簇分配
        print(f"轮数：:{iteration}")
        labels = np.zeros(X.shape[0])

        # 对每个数据点，找到最近的簇中心
        for i in range(X.shape[0]):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            labels[i] = np.argmin(distances)

        # 保存旧的簇中心
        old_centroids = centroids.copy()

        # 更新簇中心
        for k in range(K):
            points_in_cluster = X[labels == k]
            if len(points_in_cluster) > 0:
                centroids[k] = points_in_cluster.mean(axis=0)

        # 检查收敛条件
        if np.all(np.linalg.norm(centroids - old_centroids, axis=1) < tol):
            break



    return centroids, labels


# 示例数据
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# 计算凸壳
hull = ConvexHull(X)

# 绘制数据点和凸壳
plt.plot(X[:,0], X[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(X[simplex, 0], X[simplex, 1], 'k-')

plt.show()

# 判断数据集是否是凸的
is_convex = len(X) == len(hull.vertices)
print(f"数据集是否是凸的: {is_convex}")

# 运行 K-means 算法
centroids, labels = kmeans(X, K=2)

print("簇中心:", centroids)
print("簇标签:", labels)
'''

'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 创建图形和轴
fig, ax = plt.subplots(figsize=(12, 8))

# 设置标题
ax.set_title('Random Forest Algorithm', fontsize=16, weight='bold')

# 绘制原始数据集
ax.text(0.1, 0.9, 'Original Dataset', fontsize=12, weight='bold')
original_data = patches.Rectangle((0.1, 0.7), 0.15, 0.15, edgecolor='black', facecolor='lightgray')
ax.add_patch(original_data)

# 绘制Bootstrap Sampling
ax.text(0.4, 0.9, 'Bootstrap Sampling', fontsize=12, weight='bold')
for i in range(3):
    bootstrap_sample = patches.Rectangle((0.35, 0.7 - i*0.2), 0.15, 0.15, edgecolor='black', facecolor='lightgray')
    ax.add_patch(bootstrap_sample)
    ax.arrow(0.25, 0.75, 0.1, -i*0.2, head_width=0.02, head_length=0.02, fc='black', ec='black')

# 绘制训练多个决策树
ax.text(0.7, 0.9, 'Train Decision Trees', fontsize=12, weight='bold')
for i in range(3):
    tree = patches.Rectangle((0.65, 0.7 - i*0.2), 0.15, 0.15, edgecolor='black', facecolor='lightgreen')
    ax.add_patch(tree)
    ax.arrow(0.5, 0.75 - i*0.2, 0.15, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')

# 绘制Ensemble
ax.text(0.85, 0.9, 'Ensemble', fontsize=12, weight='bold')
ensemble = patches.Rectangle((0.85, 0.7), 0.1, 0.15, edgecolor='black', facecolor='lightblue')
ax.add_patch(ensemble)
for i in range(3):
    ax.arrow(0.8, 0.75 - i*0.2, 0.05, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')

# 添加注释
ax.text(0.35, 0.65, 'Bootstrap Samples', fontsize=10, color='black')
ax.text(0.65, 0.65, 'Decision Trees', fontsize=10, color='black')
ax.text(0.85, 0.65, 'Final Prediction', fontsize=10, color='black')

# 设置轴限制和隐藏轴
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 显示图形
plt.tight_layout()
plt.show()
'''

'''
import matplotlib.pyplot as plt

# 定义目标集
destroy_targets_ids = [1, 3, 4, 5, 7, 9, 10, 11, 15, 17, 18, 19]

# 数据
data_Target = [
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
    [12, 44, 66, 1, 1, 1],
    [13, 10, 60, 1, 1, 1],
    [14, -28, 84, 1, 1, 1],
    [15, -9, 51, 1, 1, 1],
    [16, 20, 80, 1, 1, 1],
    [17, -42, 25, 1, 1, 1],
    [18, 33, 46, 1, 1, 1],
    [19, 9, 32, 1, 1, 1],
    [20, -38, 70, 1, 1, 1]
]

# 提取位置信息
positions = [row[1:3] for row in data_Target]

# 遍历数据，仅绘制摧毁目标集
for i, row in enumerate(data_Target):
    if i + 1 in destroy_targets_ids:
        plt.scatter(row[1], row[2], marker='o', color='red')  # 使用红色圆圈表示
        plt.text(row[1], row[2], str(i + 1), ha='left', va='bottom')  # 在目标旁边显示序号

# 设置图例和标题
plt.legend(['Destroy Targets'])
plt.title('Destroy Targets')
plt.xlabel('X Position')
plt.ylabel('Y Position')

# 显示图形
plt.show()
'''

'''
打击次序生成，效果一般

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import distance_matrix

def create_graph(points):
    n = len(points)
    dist_matrix = distance_matrix(points, points)
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=dist_matrix[i, j])
    return G

def solve_tsp(points):
    G = create_graph(points)
    mst = nx.minimum_spanning_tree(G)
    tour = list(nx.dfs_preorder_nodes(mst, source=0))
    tour.append(tour[0])
    return tour

def plot_path(points, tour, cluster_index):
    plt.figure(figsize=(8, 6))
    for i in range(len(tour) - 1):
        plt.plot([points[tour[i], 0], points[tour[i + 1], 0]],
                 [points[tour[i], 1], points[tour[i + 1], 1]], 'b-o')
    plt.scatter(points[:, 0], points[:, 1], c='red')
    for i, point in enumerate(points):
        plt.text(point[0], point[1], str(int(cluster_index[i])), fontsize=12, ha='right')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Optimal Path for Cluster {cluster_index[0]}')
    plt.grid(True)
    plt.show()

def process_clusters(cluster):
    for i, points in enumerate(cluster):
        cluster_index = points[:, 0]
        start = np.array([0, 0])
        points_with_start = np.vstack([start, points[:, 1:3]])
        tour = solve_tsp(points_with_start)
        plot_path(points_with_start, tour, np.insert(cluster_index, 0, 0))

# Example usage
cluster = [
    np.array([
        [9., 29., 7., 1.],
        [6., 38., 21., 0.88723185],
        [21., 48., 41., 0.80221932]
    ]),
    np.array([
        [3., 35., 42., 0.7310483],
        [23., 5., 94., 0.46109201],
        [2., 1., 74., 0.44441795]
    ]),
    np.array([
        [11., -12., 15., 0.154830507],
        [1., -45., 28., 0.0120020349],
        [22., -24., 14., 0.0]
    ])
]

process_clusters(cluster)
'''

'''
打击次序生成，效果可以，但耗时

import numpy as np
import matplotlib.pyplot as plt

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def total_distance(path, points):
    dist = 0
    for i in range(len(path) - 1):
        dist += distance(points[path[i]], points[path[i + 1]])
    dist += distance(points[path[-1]], np.array([0, 0]))  # 回到原点
    return dist

class GeneticAlgorithm:
    def __init__(self, points, population_size, generations, mutation_rate, update_interval):
        self.points = points
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.update_interval = update_interval

    def create_initial_population(self):
        population = []
        for _ in range(self.population_size):
            path = np.arange(len(self.points))
            np.random.shuffle(path)
            population.append(path)
        return np.array(population)

    def fitness(self, population):
        fitness_values = []
        for path in population:
            fitness_values.append(1 / total_distance(path, self.points))
        return np.array(fitness_values)

    def selection(self, population, fitness_values):
        probs = fitness_values / np.sum(fitness_values)
        indices = np.random.choice(len(population), size=len(population), p=probs)
        return population[indices]

    def crossover(self, parent1, parent2):
        start = np.random.randint(0, len(parent1) - 1)
        end = np.random.randint(start + 1, len(parent1))
        child = np.full(len(parent1), -1)
        child[start:end] = parent1[start:end]
        remaining_indices = [i for i in range(len(parent2)) if parent2[i] not in child]
        child[child == -1] = parent2[remaining_indices]
        return child

    def mutation(self, path):
        if np.random.rand() < self.mutation_rate:
            i = np.random.randint(0, len(path))
            j = np.random.randint(0, len(path))
            path[i], path[j] = path[j], path[i]
        return path

    def evolve(self):
        population = self.create_initial_population()
        for generation in range(self.generations):
            fitness_values = self.fitness(population)
            population = self.selection(population, fitness_values)
            new_population = []
            for i in range(0, len(population), 2):
                parent1 = population[i]
                parent2 = population[i + 1]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_population.append(child1)
                new_population.append(child2)
            population = np.array(new_population)

            # 动态更新的部分
            if generation % self.update_interval == 0:
                # 在这里添加您的动态更新逻辑
                print(f"Performing dynamic update at generation {generation}")

        fitness_values = self.fitness(population)
        best_index = np.argmax(fitness_values)
        return population[best_index]

def find_optimal_path(subcluster):
    points = subcluster[:, 1:3]
    ga = GeneticAlgorithm(points, population_size=100, generations=1000, mutation_rate=0.05, update_interval=100)
    best_path = ga.evolve()
    return best_path

def plot_paths(cluster):
    plt.figure()
    plt.plot([0, 0], [0, 0], marker='o', color='red', markersize=10)  # 原点
    colors = ['green','blue','yellow']
    for i, subcluster in enumerate(cluster):
        for j, point in enumerate(subcluster[:, 1:3]):
            plt.plot(point[0], point[1], marker='o', color='blue')
            plt.text(point[0], point[1], str(subcluster[j, 0]))  # 显示标号
        best_path = find_optimal_path(subcluster)
        for k in range(len(best_path) - 1):
            plt.plot([subcluster[best_path[k], 1], subcluster[best_path[k + 1], 1]],
                     [subcluster[best_path[k], 2], subcluster[best_path[k + 1], 2]], color=colors[i])
        plt.plot([subcluster[best_path[-1], 1], 0],
                 [subcluster[best_path[-1], 2], 0], color=colors[i])  # 回到原点
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Optimal Paths')
    plt.show()

cluster =  [np.array([[ 9., 29.,  7.,  1. ],
       [ 6., 38., 21.,  0.88723185],
       [21., 48., 41.,  0.80221932]]), np.array([[ 3., 35., 42.,  0.7310483 ],
       [23.,  5., 94.,  0.46109201],
       [ 2.,  1., 74.,  0.44441795]]), np.array([[ 1.10000000e+01, -1.20000000e+01,  1.50000000e+01,
         1.54830507e-01],
       [ 1.00000000e+00, -4.50000000e+01,  2.80000000e+01,
         1.20020349e-02],
       [ 2.20000000e+01, -2.40000000e+01,  1.40000000e+01,
         0.00000000e+00]])]

plot_paths(cluster)
'''

'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 数据
data_UAV = np.array([
    [1, 0.88888889, 0.85714286, 1, 300, 1, 1],
    [2, 0.22222222, 0.14285714, 1, 350, 1, 0.94728983],
    [3, 0.77777778, 0.28571429, 1, 400, 1, 0.94905873],
    [4, 1, 0.42857143, 2, 250, 2, 0.02075731],
    [5, 0.44444444, 1, 2, 300, 2, 0.08850153],
    [6, 0.33333333, 0.42857143, 2, 350, 2, 0.09972444],
    [7, 0, 0.42857143, 3, 200, 2, 0.00675030],
    [8, 0, 0.14285714, 3, 500, 2, 0.00797593],
    [9, 0.55555556, 0.85714286, 3, 400, 2, 0.09601686],
    [10, 0.77777778, 0.14285714, 3, 300, 2, 0.10308445],
    [11, 0.11111111, 1, 3, 450, 2, 0],
    [12, 0.66666667, 0, 3, 500, 2, 0.01957874]
])

# PSO参数
num_particles = 30
num_dimensions = data_UAV.shape[0]
max_iter = 100

# 确定簇数
requested_clusters = 3
num_recon_uavs = np.sum(data_UAV[:, 3] == 1)
if requested_clusters > num_recon_uavs:
    raise ValueError(
        f"Requested number of clusters ({requested_clusters}) exceeds available reconnaissance UAVs ({num_recon_uavs}).")


# 适应度函数
def fitness_function(positions, data_UAV):
    fitness = np.zeros(num_particles)
    for i in range(num_particles):
        clusters = {k: [] for k in range(requested_clusters)}
        for j in range(num_dimensions):
            cluster_id = int(positions[i, j])
            clusters[cluster_id].append(data_UAV[j])
        fitness[i] = calculate_fitness(clusters)
    return fitness


def calculate_fitness(clusters):
    fitness = 0
    for cluster_id, uavs in clusters.items():
        if len(uavs) == 0:
            fitness += 1e6  # 惩罚空簇
            continue
        types = [uav[5] for uav in uavs]
        if 1 not in types:
            fitness += 1e6  # 惩罚没有侦察无人机的簇
        positions = np.array([[uav[1], uav[2]] for uav in uavs])
        recon_positions = positions[np.array(types) == 1]
        attack_positions = positions[np.array(types) == 2]
        if len(recon_positions) > 0:
            dists = np.min(np.linalg.norm(attack_positions[:, None] - recon_positions[None, :], axis=2), axis=1)
            fitness += np.sum(dists)  # 位置集中性
        attack_counts = [len([uav for uav in uavs if uav[5] == 2]) for uavs in clusters.values() if uavs]
        fitness += np.sum(np.abs(np.diff(sorted(attack_counts))))  # 攻击无人机数量均衡
    return fitness


# 初始化粒子
positions = np.random.randint(0, requested_clusters, (num_particles, num_dimensions))
velocities = np.random.rand(num_particles, num_dimensions)
pbest_positions = positions.copy()
pbest_fitness = fitness_function(positions, data_UAV)
gbest_position = pbest_positions[np.argmin(pbest_fitness)]
gbest_fitness = np.min(pbest_fitness)

# PSO主循环
for iter in range(max_iter):
    r1, r2 = np.random.rand(num_particles, num_dimensions), np.random.rand(num_particles, num_dimensions)
    velocities = 0.5 * velocities + r1 * (pbest_positions - positions) + r2 * (gbest_position - positions)
    positions = np.clip(positions + velocities, 0, requested_clusters - 1).astype(int)

    fitness = fitness_function(positions, data_UAV)

    better_fitness_mask = fitness < pbest_fitness
    pbest_positions[better_fitness_mask] = positions[better_fitness_mask]
    pbest_fitness[better_fitness_mask] = fitness[better_fitness_mask]

    if np.min(fitness) < gbest_fitness:
        gbest_position = positions[np.argmin(fitness)]
        gbest_fitness = np.min(fitness)

# 最优分簇结果
clusters = {k: [] for k in range(requested_clusters)}
for j in range(num_dimensions):
    cluster_id = int(gbest_position[j])
    clusters[cluster_id].append(data_UAV[j])

# 调整各簇内攻击无人机位置离散程度

# 打印最终簇分配情况
for cluster_id, uavs in clusters.items():
    print(f"Cluster {cluster_id + 1}:")
    for uav in uavs:
        print(f"UAV {int(uav[0])}: Type {int(uav[5])}, Position ({uav[1]}, {uav[2]}), Value {uav[-1]}")

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b']
for cluster_id, uavs in clusters.items():
    uavs = np.array(uavs)
    ax.scatter(uavs[:, 1], uavs[:, 2], uavs[:, -1], c=colors[cluster_id], label=f'Cluster {cluster_id + 1}')
    for uav in uavs:
        ax.text(uav[1], uav[2], uav[-1], f'{int(uav[0])}')

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Value')
plt.legend()
plt.show()
'''

'''
# If you need to import additional packages or classes, please import here.

def func():
    string = input("请输入字符串: ")
    n = int(input("请输入字符串表的长度: "))
    word_list = []
    for _ in range(n):
        word_list.append(input("请输入字符串表中的单词: "))

    def find_max_word_sequence(start_index):
        max_count = 0
        for i in range(start_index, len(string)):
            for j in range(i + 1, len(string) + 1):
                sub_string = string[i:j]
                if sub_string in word_list:
                    count = find_max_word_sequence(j) + 1
                    if count > max_count:
                        max_count = count
        return max_count

    print(find_max_word_sequence(0))

if __name__ == "__main__":
    func()
'''

# If you need to import additional packages or classes, please import here.
'''
def func():
    k = int(input())
    fragments = []
    for _ in range(k):
        start, end = map(int, input().split())
        fragments.append((start, end))

    def is_overlapping(frag1, frag2):
        return not (frag1[1] < frag2[0] or frag2[1] < frag1[0])

    def merge_fragments(frag1, frag2):
        return (min(frag1[0], frag2[0]), max(frag1[1], frag2[1]))

    def find_min_fragments(fragments):  # 在这里添加参数 fragments
        while len(fragments) > 1:
            new_fragments = []
            merged = False
            for i in range(len(fragments)):
                for j in range(i + 1, len(fragments)):
                    if is_overlapping(fragments[i], fragments[j]):
                        new_fragment = merge_fragments(fragments[i], fragments[j])
                        new_fragments.append(new_fragment)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                new_fragments.append(fragments.pop())
            else:
                fragments = new_fragments
        return len(fragments)

    print(find_min_fragments(fragments))  # 在这里将 fragments 作为参数传递

if __name__ == "__main__":
    func()
'''
