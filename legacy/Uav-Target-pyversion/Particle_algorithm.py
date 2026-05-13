"""历史版本中的particlealgorithm脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np

"""
class PSO:
    def __init__(self, num_particles, num_targets, num_clusters, w=0.5, c1=1, c2=2, max_iter=100, seed = None):
        if seed is not None:
            np.random.seed(seed)
        self.num_particles = num_particles
        self.num_targets = num_targets
        self.num_clusters = num_clusters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

        self.particles = np.random.randint(0, num_clusters, (num_particles, num_targets))
        self.velocities = np.random.uniform(-1, 1, (num_particles, num_targets))
        self.pbest = self.particles.copy()
        self.pbest_scores = np.full(num_particles, float('inf'))
        self.gbest = None
        self.gbest_score = float('inf')


    def calculate_angle_penalty(self, clusters):
        angle_penalty = 0
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            angles = np.array([target[5] for target in cluster])
            angle_diff = np.abs(np.subtract.outer(angles, angles))
            avg_angle_diff = np.mean(angle_diff)
            angle_penalty += avg_angle_diff
        return angle_penalty

    def fitness(self, particle, targets):
        clusters = [[] for _ in range(self.num_clusters)]
        for idx, cluster_id in enumerate(particle):
            clusters[cluster_id].append(targets[idx])


        total_value = sum(sum(target[2] for target in cluster) for cluster in clusters)


        cluster_sizes = [len(cluster) for cluster in clusters]
        max_size = max(cluster_sizes)
        min_size = min(cluster_sizes)
        balance_penalty = max_size - min_size


        angle_penalty = self.calculate_angle_penalty(clusters)


        angle_similarity_penalty = 0
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            angles = np.array([target[5] for target in cluster])
            angle_diff = np.abs(np.subtract.outer(angles, angles))
            avg_angle_diff = np.mean(angle_diff)
            angle_similarity_penalty += avg_angle_diff

        return balance_penalty + angle_penalty


    def update_particles(self, targets):
        for i in range(self.num_particles):
            fitness_value = self.fitness(self.particles[i], targets)
            if fitness_value < self.pbest_scores[i]:
                self.pbest_scores[i] = fitness_value
                self.pbest[i] = self.particles[i].copy()
            if fitness_value < self.gbest_score:
                self.gbest_score = fitness_value
                self.gbest = self.particles[i].copy()

        for i in range(self.num_particles):
            r1 = np.random.rand(self.num_targets)
            r2 = np.random.rand(self.num_targets)
            self.velocities[i] = (self.w * self.velocities[i] +
                                  self.c1 * r1 * (self.pbest[i] - self.particles[i]) +
                                  self.c2 * r2 * (self.gbest - self.particles[i]))
            self.particles[i] = np.clip(self.particles[i] + self.velocities[i], 0, self.num_clusters - 1).astype(int)

    def optimize(self, targets):

        angle_intervals = np.linspace(0, 180, self.num_clusters + 1)
        initial_particle = np.zeros(self.num_targets, dtype=int)
        for i, target in enumerate(targets):
            angle = target[5]
            for j in range(len(angle_intervals) - 1):
                if angle_intervals[j] <= angle < angle_intervals[j + 1]:
                    initial_particle[i] = j
                    break
        self.particles = np.tile(initial_particle, (self.num_particles, 1))

        for _ in range(self.max_iter):
            self.update_particles(targets)


            cluster_sizes = [len([target for target in self.particles[0] if target == i]) for i in
                             range(self.num_clusters)]
            avg_size = self.num_targets / self.num_clusters
            for i in range(self.num_clusters - 1):
                if cluster_sizes[i] > avg_size + 1:
                    angle_intervals[i + 1] -= 1
                elif cluster_sizes[i] < avg_size - 1:
                    angle_intervals[i + 1] += 1

        return self.gbest, self.gbest_score


def pso_optimization(targets, num_clusters, num_particles=30, w=0.5, c1=1, c2=2, max_iter=100):
    pso = PSO(num_particles=num_particles, num_targets=len(targets), num_clusters=num_clusters, w=w, c1=c1, c2=c2,
              max_iter=max_iter)
    best_solution, best_value = pso.optimize(targets)
    return best_solution, best_value
"""

'''
class PSO:
    def __init__(self, num_particles, num_targets, num_clusters, w=0.5, c1=1, c2=2, max_iter=100, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.num_particles = num_particles
        self.num_targets = num_targets
        self.num_clusters = num_clusters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter


        self.particles = np.random.uniform(0, 180, (num_particles, num_clusters - 1))
        self.velocities = np.random.uniform(-1, 1, (num_particles, num_clusters - 1))
        self.pbest = self.particles.copy()
        self.pbest_scores = np.full(num_particles, float('inf'))
        self.gbest = None
        self.gbest_score = float('inf')

    def initial_classification(self, targets):
        clusters = [[] for _ in range(self.num_clusters)]
        angles = targets[:, 5]
        angle_intervals = np.linspace(0, 180, self.num_clusters + 1)
        for i, angle in enumerate(angles):
            for j in range(self.num_clusters):
                if angle_intervals[j] <= angle < angle_intervals[j + 1]:
                    clusters[j].append(targets[i])
                    break
        return clusters

    def fitness(self, particle, targets):
        angle_intervals = np.concatenate(([0], np.sort(particle), [180]))
        clusters = [[] for _ in range(self.num_clusters)]
        for target in targets:
            angle = target[5]
            for j in range(self.num_clusters):
                if angle_intervals[j] <= angle < angle_intervals[j + 1]:
                    clusters[j].append(target)
                    break

        cluster_sizes = [len(cluster) for cluster in clusters]
        max_size = max(cluster_sizes)
        min_size = min(cluster_sizes)
        balance_penalty = max_size - min_size

        return balance_penalty

    def update_particles(self, targets):
        for i in range(self.num_particles):
            fitness_value = self.fitness(self.particles[i], targets)
            if fitness_value < self.pbest_scores[i]:
                self.pbest_scores[i] = fitness_value
                self.pbest[i] = self.particles[i].copy()
            if fitness_value < self.gbest_score:
                self.gbest_score = fitness_value
                self.gbest = self.particles[i].copy()

        for i in range(self.num_particles):
            r1 = np.random.rand(self.num_clusters - 1)
            r2 = np.random.rand(self.num_clusters - 1)
            self.velocities[i] = (self.w * self.velocities[i] +
                                  self.c1 * r1 * (self.pbest[i] - self.particles[i]) +
                                  self.c2 * r2 * (self.gbest - self.particles[i]))
            self.particles[i] = np.clip(self.particles[i] + self.velocities[i], 0, 180)

    def optimize(self, targets):
        for _ in range(self.max_iter):
            self.update_particles(targets)
        return self.gbest, self.gbest_score

    def classify_targets(self, targets, best_solution):
        angle_intervals = np.concatenate(([0], np.sort(best_solution), [180]))
        clusters = [[] for _ in range(self.num_clusters)]
        for target in targets:
            angle = target[5]
            for j in range(self.num_clusters):
                if angle_intervals[j] <= angle < angle_intervals[j + 1]:
                    clusters[j].append(target)
                    break
        return clusters


def pso_optimization(targets, num_clusters, num_particles=30, w=0.5, c1=1, c2=2, max_iter=100):
    pso = PSO(num_particles=num_particles, num_targets=len(targets), num_clusters=num_clusters, w=w, c1=c1, c2=c2, max_iter=max_iter)
    best_solution, best_value = pso.optimize(targets)
    clusters = pso.classify_targets(targets, best_solution)
    return best_solution, best_value, clusters



targets = np.array([[1.00000000e+00, 4.30107527e-02, 2.35955056e-01, 1.00000000e+00, 4.39767448e-01, 1.48109208e+02, 4.00000000e+00],
                    [2.00000000e+00, 5.37634409e-01, 7.52808989e-01, 7.67193659e-01, 7.13159463e-01, 8.92257798e+01, 4.00000000e+00],
                    [3.00000000e+00, 9.03225806e-01, 3.93258427e-01, 9.51070910e-01, 4.61524376e-01, 5.01944289e+01, 3.00000000e+00],
                    [6.00000000e+00, 9.35483871e-01, 1.57303371e-01, 4.07307170e-01, 3.15044306e-01, 2.89264258e+01, 2.00000000e+00],
                    [9.00000000e+00, 8.38709677e-01, 0.00000000e+00, 5.16145326e-01, 1.38259266e-01, 1.35704344e+01, 2.00000000e+00],
                    [1.10000000e+01, 3.97849462e-01, 8.98876404e-02, 4.92144504e-01, 0.00000000e+00, 1.28659808e+02, 1.00000000e+00],
                    [1.70000000e+01, 7.52688172e-02, 2.02247191e-01, 3.71663739e-01, 3.86113922e-01, 1.49237280e+02, 1.00000000e+00],
                    [1.90000000e+01, 6.23655914e-01, 2.80898876e-01, 3.62766634e-01, 1.82621366e-01, 7.42913622e+01, 1.00000000e+00],
                    [2.20000000e+01, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00, 3.56302322e-01, 8.13010235e+00, 2.00000000e+00]])

num_clusters = 3
best_solution, best_value, clusters = pso_optimization(targets, num_clusters)

print("Best Solution (angle boundaries):", best_solution)
print("Best Fitness Value:", best_value)
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}:")
    for target in cluster:
        print(target)
'''

import numpy as np

class PSO:
    """PSO 类，封装PSO 算法相关的数据结构与业务行为。

    属性：
        num_particles: numparticles。
        num_targets: num目标集合。
        num_clusters: num目标簇集合。
        w: w 数据。
        c1: c1 数据。
        c2: c2 数据。
        max_iter: 最大值iter。
        particles: particles 数据。
        velocities: velocities 数据。
        pbest: pbest 数据。
        pbest_scores: pbest评分集合。
        gbest: gbest 数据。
        gbest_score: gbest评分。
    """
    def __init__(self, num_particles, num_targets, num_clusters, w=0.5, c1=1, c2=2, max_iter=100, seed=None):
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            num_particles: num_particles 参数。
            num_targets: num_targets 参数。
            num_clusters: num_clusters 参数。
            w: w 参数。
            c1: c1 参数。
            c2: c2 参数。
            max_iter: max_iter 参数。
            seed: 随机种子。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        if seed is not None:
            np.random.seed(seed)
        # num_particles: numparticles。
        self.num_particles = num_particles
        # num_targets: num目标集合。
        self.num_targets = num_targets
        # num_clusters: num目标簇集合。
        self.num_clusters = num_clusters
        # w: w 数据。
        self.w = w
        # c1: c1 数据。
        self.c1 = c1
        # c2: c2 数据。
        self.c2 = c2
        # max_iter: 最大值iter。
        self.max_iter = max_iter


        # particles: particles 数据。
        self.particles = np.random.uniform(0, 180, (num_particles, num_clusters - 1))
        # velocities: velocities 数据。
        self.velocities = np.random.uniform(-1, 1, (num_particles, num_clusters - 1))
        # pbest: pbest 数据。
        self.pbest = self.particles.copy()
        # pbest_scores: pbest评分集合。
        self.pbest_scores = np.full(num_particles, float('inf'))
        # gbest: gbest 数据。
        self.gbest = None
        # gbest_score: gbest评分。
        self.gbest_score = float('inf')

    def fitness(self, particle, targets):
        """处理fitness 数据相关业务逻辑。

        参数：
            particle: particle 数据。
            targets: 目标集合。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        angle_intervals = np.concatenate(([0], np.sort(particle), [180]))
        clusters = [[] for _ in range(self.num_clusters)]
        for target in targets:
            angle = target[5]
            for j in range(self.num_clusters):
                if angle_intervals[j] <= angle < angle_intervals[j + 1]:
                    clusters[j].append(target)
                    break

        cluster_sizes = [len(cluster) for cluster in clusters]
        max_size = max(cluster_sizes)
        min_size = min(cluster_sizes)
        balance_penalty = max_size - min_size

        return balance_penalty

    def update_particles(self, targets):
        """处理updateparticles相关业务逻辑。

        参数：
            targets: 目标集合。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        for i in range(self.num_particles):
            fitness_value = self.fitness(self.particles[i], targets)
            if fitness_value < self.pbest_scores[i]:
                self.pbest_scores[i] = fitness_value
                self.pbest[i] = self.particles[i].copy()
            if fitness_value < self.gbest_score:
                self.gbest_score = fitness_value
                self.gbest = self.particles[i].copy()

        for i in range(self.num_particles):
            r1 = np.random.rand(self.num_clusters - 1)
            r2 = np.random.rand(self.num_clusters - 1)
            self.velocities[i] = (self.w * self.velocities[i] +
                                  self.c1 * r1 * (self.pbest[i] - self.particles[i]) +
                                  self.c2 * r2 * (self.gbest - self.particles[i]))
            self.particles[i] = np.clip(self.particles[i] + self.velocities[i], 0, 180)

    def optimize(self, targets):
        """处理optimize 数据相关业务逻辑。

        参数：
            targets: 目标集合。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        for _ in range(self.max_iter):
            self.update_particles(targets)
        return self.gbest, self.gbest_score

    def classify_targets(self, targets, best_solution):
        """处理classify目标集合相关业务逻辑。

        参数：
            targets: 目标集合。
            best_solution: bestsolution。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        angle_intervals = np.concatenate(([0], np.sort(best_solution), [180]))
        clusters = [[] for _ in range(self.num_clusters)]
        best_assignment = np.zeros(len(targets), dtype=int)
        for i, target in enumerate(targets):
            angle = target[5]
            for j in range(self.num_clusters):
                if angle_intervals[j] <= angle < angle_intervals[j + 1]:
                    clusters[j].append(target)
                    best_assignment[i] = j
                    break
        return clusters, best_assignment


def pso_optimization(targets, num_clusters, num_particles=30, w=0.5, c1=1, c2=2, max_iter=100):
    """处理PSO 算法optimization相关业务逻辑。

    参数：
        targets: 目标集合。
        num_clusters: num目标簇集合。
        num_particles: numparticles。
        w: w 数据。
        c1: c1 数据。
        c2: c2 数据。
        max_iter: 最大值iter。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    pso = PSO(num_particles=num_particles, num_targets=len(targets), num_clusters=num_clusters, w=w, c1=c1, c2=c2, max_iter=max_iter)
    best_solution, best_value = pso.optimize(targets)
    clusters, best_assignment = pso.classify_targets(targets, best_solution)
    return best_solution, best_value, best_assignment


targets = np.array([[1.00000000e+00, 4.30107527e-02, 2.35955056e-01, 1.00000000e+00, 4.39767448e-01, 1.48109208e+02, 4.00000000e+00],
                    [2.00000000e+00, 5.37634409e-01, 7.52808989e-01, 7.67193659e-01, 7.13159463e-01, 8.92257798e+01, 4.00000000e+00],
                    [3.00000000e+00, 9.03225806e-01, 3.93258427e-01, 9.51070910e-01, 4.61524376e-01, 5.01944289e+01, 3.00000000e+00],
                    [6.00000000e+00, 9.35483871e-01, 1.57303371e-01, 4.07307170e-01, 3.15044306e-01, 2.89264258e+01, 2.00000000e+00],
                    [9.00000000e+00, 8.38709677e-01, 0.00000000e+00, 5.16145326e-01, 1.38259266e-01, 1.35704344e+01, 2.00000000e+00],
                    [1.10000000e+01, 3.97849462e-01, 8.98876404e-02, 4.92144504e-01, 0.00000000e+00, 1.28659808e+02, 1.00000000e+00],
                    [1.70000000e+01, 7.52688172e-02, 2.02247191e-01, 3.71663739e-01, 3.86113922e-01, 1.49237280e+02, 1.00000000e+00],
                    [1.90000000e+01, 6.23655914e-01, 2.80898876e-01, 3.62766634e-01, 1.82621366e-01, 7.42913622e+01, 1.00000000e+00],
                    [2.20000000e+01, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00, 3.56302322e-01, 8.13010235e+00, 2.00000000e+00]])

num_clusters = 3
best_solution, best_value, best_assignment = pso_optimization(targets, num_clusters)

print("Best Solution (angle boundaries):", best_solution)
print("Best Fitness Value:", best_value)
print("Best Assignment:", best_assignment)


