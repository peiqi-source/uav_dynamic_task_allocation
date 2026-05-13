"""资源分配模块中的PSO 算法clusterer实现。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from uav_dynamic_task_allocation.allocation.clustering_metrics import (
    pso_objective_metrics,
)


class PSOClustererError(Exception):
    """Raised when PSO target clustering fails."""


@dataclass(frozen=True)
class PSOClustererConfig:
    """Configuration for PSO-based target region clustering."""

    # num_particles: numparticles。
    num_particles: int = 30
    # max_iter: 最大值iter。
    max_iter: int = 100
    # inertia_weight: inertia权重。
    inertia_weight: float = 0.7
    # cognitive_weight: cognitive权重。
    cognitive_weight: float = 1.5
    # social_weight: social权重。
    social_weight: float = 1.5
    # compactness_weight: compactness权重。
    compactness_weight: float = 1.0
    # balance_weight: balance权重。
    balance_weight: float = 0.1
    # value_weight: 数值权重。
    value_weight: float = 0.0
    # defense_weight: 防御能力权重。
    defense_weight: float = 0.0
    # convergence_threshold: convergencethreshold。
    convergence_threshold: float = 1e-6
    # random_seed: 随机随机种子。
    random_seed: int = 42
    # initialize_with_high_value_targets: initializewithhigh数值目标集合。
    initialize_with_high_value_targets: bool = True

    def validate(self) -> None:
        """校验当前对象或输入配置的合法性。

        参数：
            无显式业务参数。

        返回：
            无返回值；通过状态变更、文件输出或日志记录体现执行结果。
        """
        if self.num_particles <= 0:
            raise PSOClustererError("num_particles must be positive.")
        if self.max_iter <= 0:
            raise PSOClustererError("max_iter must be positive.")
        if self.convergence_threshold < 0:
            raise PSOClustererError("convergence_threshold must be non-negative.")


@dataclass
class PSOClustererResult:
    """Output of the PSO cluster-center optimization."""

    # labels: labels 数据。
    labels: np.ndarray
    # centers: centers 数据。
    centers: np.ndarray
    # best_score: best评分。
    best_score: float
    # convergence_history: convergencehistory。
    convergence_history: list[float] = field(default_factory=list)
    # metrics: 指标集合。
    metrics: dict[str, float] = field(default_factory=dict)
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)


class PSOClusterer:
    """
    Particle Swarm Optimization clusterer.

    Each particle is a flattened matrix of cluster centers in normalized feature
    space. The objective favors compact clusters and balanced target counts,
    with optional value/defense weighting supplied per target.
    """

    def __init__(self, config: PSOClustererConfig) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            config: 配置对象，类型为 PSOClustererConfig。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        config.validate()
        # config: 配置。
        self.config = config

    def cluster(
        self,
        features: np.ndarray,
        num_clusters: int,
        target_values: np.ndarray | None = None,
        target_defenses: np.ndarray | None = None,
    ) -> PSOClustererResult:
        """处理目标簇相关业务逻辑。

        参数：
            features: features 数据，类型为 np.ndarray。
            num_clusters: num目标簇集合，类型为 int。
            target_values: 目标values，类型为 np.ndarray | None。
            target_defenses: 目标defenses，类型为 np.ndarray | None。

        返回：
            PSOClustererResult，表示该函数计算或构建得到的结果。
        """
        if features.ndim != 2:
            raise PSOClustererError("features must be a 2D array.")
        if num_clusters <= 0:
            raise PSOClustererError("num_clusters must be positive.")
        if len(features) < num_clusters:
            raise PSOClustererError("num samples must be >= num_clusters.")

        rng = np.random.default_rng(self.config.random_seed)
        num_samples, feature_dim = features.shape
        particle_shape = (num_clusters, feature_dim)

        positions = rng.random(
            size=(self.config.num_particles, *particle_shape)
        )
        if self.config.initialize_with_high_value_targets and target_values is not None:
            top_indices = np.argsort(np.asarray(target_values))[-num_clusters:]
            positions[0] = features[top_indices]

        velocities = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(self.config.num_particles, *particle_shape),
        )

        personal_best_positions = positions.copy()
        personal_best_scores = np.asarray(
            [
                self._objective(
                    features=features,
                    centers=positions[index],
                    target_values=target_values,
                    target_defenses=target_defenses,
                )
                for index in range(self.config.num_particles)
            ],
            dtype=np.float64,
        )

        best_index = int(personal_best_scores.argmin())
        global_best = personal_best_positions[best_index].copy()
        global_best_score = float(personal_best_scores[best_index])
        history = [global_best_score]

        stale_iterations = 0

        for _ in range(self.config.max_iter):
            previous_best = global_best_score

            for particle_index in range(self.config.num_particles):
                r1 = rng.random(size=particle_shape)
                r2 = rng.random(size=particle_shape)

                velocities[particle_index] = (
                    self.config.inertia_weight * velocities[particle_index]
                    + self.config.cognitive_weight
                    * r1
                    * (personal_best_positions[particle_index] - positions[particle_index])
                    + self.config.social_weight
                    * r2
                    * (global_best - positions[particle_index])
                )

                positions[particle_index] = np.clip(
                    positions[particle_index] + velocities[particle_index],
                    0.0,
                    1.0,
                )

                score = self._objective(
                    features=features,
                    centers=positions[particle_index],
                    target_values=target_values,
                    target_defenses=target_defenses,
                )

                if score < personal_best_scores[particle_index]:
                    personal_best_scores[particle_index] = score
                    personal_best_positions[particle_index] = positions[
                        particle_index
                    ].copy()

                    if score < global_best_score:
                        global_best_score = float(score)
                        global_best = positions[particle_index].copy()

            history.append(global_best_score)

            if abs(previous_best - global_best_score) <= self.config.convergence_threshold:
                stale_iterations += 1
                if stale_iterations >= 5:
                    break
            else:
                stale_iterations = 0

        labels = self._assign_labels(features, global_best)
        centers = self._recompute_centers(features, labels, global_best)
        final_score = self._objective(
            features=features,
            centers=centers,
            target_values=target_values,
            target_defenses=target_defenses,
        )
        labels = self._assign_labels(features, centers)

        return PSOClustererResult(
            labels=labels,
            centers=centers,
            best_score=float(final_score),
            convergence_history=history,
            metrics=pso_objective_metrics(features, labels, centers),
            metadata={
                "num_particles": self.config.num_particles,
                "max_iter": self.config.max_iter,
                "num_iter": len(history) - 1,
                "num_clusters": num_clusters,
                "feature_dim": feature_dim,
            },
        )

    def _objective(
        self,
        features: np.ndarray,
        centers: np.ndarray,
        target_values: np.ndarray | None,
        target_defenses: np.ndarray | None,
    ) -> float:
        """处理objective 数据相关业务逻辑。

        参数：
            features: features 数据，类型为 np.ndarray。
            centers: centers 数据，类型为 np.ndarray。
            target_values: 目标values，类型为 np.ndarray | None。
            target_defenses: 目标defenses，类型为 np.ndarray | None。

        返回：
            float，表示该函数计算或构建得到的结果。
        """
        labels = self._assign_labels(features, centers)
        distances = np.linalg.norm(features - centers[labels], axis=1)

        sample_weights = np.ones(len(features), dtype=np.float64)
        if target_values is not None and self.config.value_weight > 0:
            values = self._normalize_vector(target_values)
            sample_weights += self.config.value_weight * values
        if target_defenses is not None and self.config.defense_weight > 0:
            defenses = self._normalize_vector(target_defenses)
            sample_weights += self.config.defense_weight * defenses

        compactness = float(np.average(distances, weights=sample_weights))
        counts = np.asarray(
            [np.sum(labels == cluster_id) for cluster_id in range(len(centers))],
            dtype=np.float64,
        )
        ideal = len(features) / len(centers)
        balance_penalty = float(np.mean((counts - ideal) ** 2) / max(ideal, 1.0))
        empty_penalty = float(np.sum(counts == 0)) * 10.0

        return (
            self.config.compactness_weight * compactness
            + self.config.balance_weight * balance_penalty
            + empty_penalty
        )

    @staticmethod
    def _assign_labels(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """处理assignlabels相关业务逻辑。

        参数：
            features: features 数据，类型为 np.ndarray。
            centers: centers 数据，类型为 np.ndarray。

        返回：
            np.ndarray，表示该函数计算或构建得到的结果。
        """
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        return distances.argmin(axis=1).astype(np.int64)

    @staticmethod
    def _recompute_centers(
        features: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        """处理recomputecenters相关业务逻辑。

        参数：
            features: features 数据，类型为 np.ndarray。
            labels: labels 数据，类型为 np.ndarray。
            centers: centers 数据，类型为 np.ndarray。

        返回：
            np.ndarray，表示该函数计算或构建得到的结果。
        """
        new_centers = centers.copy()
        for cluster_id in range(len(centers)):
            members = features[labels == cluster_id]
            if len(members) > 0:
                new_centers[cluster_id] = members.mean(axis=0)
        return new_centers

    @staticmethod
    def _normalize_vector(values: np.ndarray) -> np.ndarray:
        """处理normalizevector相关业务逻辑。

        参数：
            values: values 数据，类型为 np.ndarray。

        返回：
            np.ndarray，表示该函数计算或构建得到的结果。
        """
        values = np.asarray(values, dtype=np.float64)
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if abs(max_value - min_value) < 1e-12:
            return np.zeros_like(values, dtype=np.float64)
        return (values - min_value) / (max_value - min_value)
