"""资源分配模块中的clustering指标集合实现。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from uav_dynamic_task_allocation.core.contracts import TargetClusterSet


class ClusteringMetricsError(Exception):
    """Raised when clustering metrics cannot be computed."""


@dataclass(frozen=True)
class ClusteringMetrics:
    """Compact summary of clustering quality."""

    # num_clusters: num目标簇集合。
    num_clusters: int
    # num_targets: num目标集合。
    num_targets: int
    # mean_compactness: 均值compactness。
    mean_compactness: float
    # max_compactness: 最大值compactness。
    max_compactness: float
    # target_count_std: 目标count标准差。
    target_count_std: float
    # target_count_balance_score: 目标countbalance评分。
    target_count_balance_score: float
    # workload_std: workload标准差。
    workload_std: float
    # workload_balance_score: workloadbalance评分。
    workload_balance_score: float

    def to_dict(self) -> dict[str, Any]:
        """将对象转换为字典，便于日志记录、序列化或调试输出。

        参数：
            无显式业务参数。

        返回：
            dict[str, Any]，表示该函数计算或构建得到的结果。
        """
        return {
            "num_clusters": self.num_clusters,
            "num_targets": self.num_targets,
            "mean_compactness": self.mean_compactness,
            "max_compactness": self.max_compactness,
            "target_count_std": self.target_count_std,
            "target_count_balance_score": self.target_count_balance_score,
            "workload_std": self.workload_std,
            "workload_balance_score": self.workload_balance_score,
        }


def calculate_target_count_balance(counts: list[int]) -> tuple[float, float]:
    """Return target-count standard deviation and a normalized balance score."""
    if not counts:
        return 0.0, 0.0

    values = np.asarray(counts, dtype=np.float64)
    std = float(np.std(values))
    mean = float(np.mean(values))
    balance_score = 1.0 / (1.0 + std / max(mean, 1e-8))
    return std, float(balance_score)


def calculate_workload_balance(workloads: list[float]) -> tuple[float, float]:
    """Return workload standard deviation and a normalized balance score."""
    if not workloads:
        return 0.0, 0.0

    values = np.asarray(workloads, dtype=np.float64)
    std = float(np.std(values))
    mean = float(np.mean(values))
    balance_score = 1.0 / (1.0 + std / max(mean, 1e-8))
    return std, float(balance_score)


def evaluate_cluster_set(cluster_set: TargetClusterSet) -> ClusteringMetrics:
    """Compute paper-facing clustering metrics from a TargetClusterSet."""
    cluster_set.validate()

    compactness_values = [
        float(cluster.compactness or 0.0)
        for cluster in cluster_set.clusters
    ]
    counts = [cluster.num_targets for cluster in cluster_set.clusters]
    workloads = [
        float(cluster.defense_sum) + float(cluster.significance_sum)
        for cluster in cluster_set.clusters
    ]

    target_count_std, target_count_balance = calculate_target_count_balance(counts)
    workload_std, workload_balance = calculate_workload_balance(workloads)

    return ClusteringMetrics(
        num_clusters=cluster_set.num_clusters,
        num_targets=len(cluster_set.all_target_ids),
        mean_compactness=float(np.mean(compactness_values)) if compactness_values else 0.0,
        max_compactness=float(np.max(compactness_values)) if compactness_values else 0.0,
        target_count_std=target_count_std,
        target_count_balance_score=target_count_balance,
        workload_std=workload_std,
        workload_balance_score=workload_balance,
    )


def pso_objective_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> dict[str, float]:
    """Compute low-level metrics for normalized feature-space clustering."""
    if features.ndim != 2 or centers.ndim != 2:
        raise ClusteringMetricsError("features and centers must be 2D arrays.")

    if len(features) != len(labels):
        raise ClusteringMetricsError("labels length must match number of samples.")

    distances = np.linalg.norm(features - centers[labels], axis=1)
    counts = [int(np.sum(labels == cluster_id)) for cluster_id in range(len(centers))]
    target_count_std, target_count_balance = calculate_target_count_balance(counts)

    return {
        "mean_distance_to_center": float(np.mean(distances)) if len(distances) else 0.0,
        "max_distance_to_center": float(np.max(distances)) if len(distances) else 0.0,
        "target_count_std": target_count_std,
        "target_count_balance_score": target_count_balance,
    }
