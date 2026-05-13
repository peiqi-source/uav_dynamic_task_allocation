from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from uav_dynamic_task_allocation.allocation.clustering_metrics import (
    evaluate_cluster_set,
)
from uav_dynamic_task_allocation.allocation.pso_clusterer import (
    PSOClusterer,
    PSOClustererConfig,
)
from uav_dynamic_task_allocation.allocation.ppo_clusterer import (
    PPOClusterer,
    PPOClustererConfig,
)
from uav_dynamic_task_allocation.core.contracts import (
    AlgorithmMetadata,
    PipelineStage,
    ScreenedTargetSet,
    TargetCluster,
    TargetClusterSet,
)
from uav_dynamic_task_allocation.core.entities import Position, Target
from uav_dynamic_task_allocation.utils.config import get_config_value, resolve_path


class TargetClusteringError(Exception):
    """目标分群过程中的自定义错误。"""


@dataclass(frozen=True)
class TargetClusteringConfig:
    """
    目标分群配置。

    该模块用于将 destroy_targets 划分成若干目标群。
    当前支持：
    - kmeans：稳定、可解释的工程 baseline；
    - pso：对齐源代码 PSO_classify 的群中心优化思想。

    后续如果加入 PPO 分群，只需要新增一个 _cluster_by_ppo 分支，
    并保持输出 TargetClusterSet 不变，后续资源分配和 DQN 排序无需修改。
    """

    method: str = "kmeans"
    num_clusters: int = 3
    auto_adjust_num_clusters: bool = True
    feature_names: tuple[str, ...] = ("x", "y")

    # KMeans
    kmeans_max_iter: int = 300
    kmeans_tolerance: float = 0.0004
    kmeans_random_seed: int = 42

    # PSO
    pso_num_particles: int = 30
    pso_max_iter: int = 100
    pso_inertia_weight: float = 0.7
    pso_cognitive_weight: float = 1.5
    pso_social_weight: float = 1.5
    pso_compactness_weight: float = 1.0
    pso_balance_weight: float = 0.1
    pso_value_weight: float = 0.2
    pso_defense_weight: float = 0.2
    pso_convergence_threshold: float = 1e-6
    pso_initialize_with_high_value_targets: bool = True
    pso_random_seed: int = 42
    ppo_checkpoint_dir: str = "checkpoints/ppo_clusterer"
    ppo_fallback_method: str = "pso"

    debug_csv_path: str = "outputs/intermediate/target_clustering.csv"

    def validate(self) -> None:
        """检查配置是否合法。"""
        supported_methods = {"kmeans", "pso", "ppo"}

        if self.method not in supported_methods:
            raise TargetClusteringError(
                f"Unsupported target clustering method={self.method}. "
                f"Supported methods: {sorted(supported_methods)}"
            )

        if self.num_clusters <= 0:
            raise TargetClusteringError("num_clusters must be positive.")

        if not self.feature_names:
            raise TargetClusteringError("feature_names must not be empty.")

        if self.kmeans_max_iter <= 0:
            raise TargetClusteringError("kmeans_max_iter must be positive.")

        if self.kmeans_tolerance <= 0:
            raise TargetClusteringError("kmeans_tolerance must be positive.")

        if self.pso_num_particles <= 0:
            raise TargetClusteringError("pso_num_particles must be positive.")

        if self.pso_max_iter <= 0:
            raise TargetClusteringError("pso_max_iter must be positive.")


@dataclass
class TargetClusteringRecord:
    """
    单个目标的分群记录。

    用于 debug CSV 和后续分析。
    """

    target: Target
    cluster_id: int
    feature_vector: list[float]
    distance_to_cluster_center: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetClusteringResult:
    """
    目标分群结果。

    cluster_set:
        标准化输出，后续 resource_allocation.py 会直接使用。

    records:
        每个目标对应的 cluster_id 和距离信息。

    metadata:
        保存算法迭代次数、目标函数值等调试信息。
    """

    source_screened_set: ScreenedTargetSet
    cluster_set: TargetClusterSet
    records: list[TargetClusteringRecord]
    metadata: dict[str, Any] = field(default_factory=dict)


class TargetClusterer:
    """
    目标分群器。

    输入：
        ScreenedTargetSet，其中 selected_targets 应该已经是摧毁目标集。

    输出：
        TargetClusterSet。

    设计原则：
        目标分群算法可以替换，但输出合同不变。
    """

    def __init__(self, config: TargetClusteringConfig) -> None:
        config.validate()
        self.config = config

    def cluster(
        self,
        screened_set: ScreenedTargetSet,
    ) -> TargetClusteringResult:
        """
        对摧毁目标集进行目标分群。
        """
        screened_set.validate()

        targets = screened_set.selected_targets

        if not targets:
            raise TargetClusteringError(
                "screened_set.selected_targets must not be empty. "
                "Please run destroy_target_selection before target_clustering."
            )

        actual_num_clusters = self._resolve_num_clusters(
            num_targets=len(targets),
            requested_num_clusters=self.config.num_clusters,
        )

        features, target_ids = self._build_feature_matrix(
            targets=targets,
            screened_set=screened_set,
        )

        normalized_features, feature_stats = self._normalize_features(features)

        if self.config.method == "kmeans":
            labels, centers_norm, metadata = self._cluster_by_kmeans(
                normalized_features=normalized_features,
                num_clusters=actual_num_clusters,
            )

        elif self.config.method == "pso":
            labels, centers_norm, metadata = self._cluster_by_pso(
                normalized_features=normalized_features,
                num_clusters=actual_num_clusters,
                targets=targets,
            )
        elif self.config.method == "ppo":
            labels, centers_norm, metadata = self._cluster_by_ppo(
                normalized_features=normalized_features,
                num_clusters=actual_num_clusters,
                targets=targets,
            )

        else:
            raise TargetClusteringError(
                f"Unsupported clustering method: {self.config.method}"
            )

        centers = self._denormalize_features(
            normalized_features=centers_norm,
            feature_stats=feature_stats,
        )

        cluster_set, records = self._build_cluster_set_and_records(
            targets=targets,
            target_ids=target_ids,
            features=features,
            labels=labels,
            centers=centers,
            screened_set=screened_set,
            algorithm_metadata=AlgorithmMetadata(
                algorithm_name=f"target_clustering_{self.config.method}",
                algorithm_type=self.config.method,
                stage=PipelineStage.TARGET_CLUSTERING,
                version="v1",
                config={
                    "method": self.config.method,
                    "requested_num_clusters": self.config.num_clusters,
                    "actual_num_clusters": actual_num_clusters,
                    "feature_names": list(self.config.feature_names),
                },
                notes=(
                    "Target clustering module. "
                    "It receives destroy targets and outputs TargetClusterSet "
                    "for resource allocation and strike-order planning."
                ),
            ),
        )
        cluster_metrics = evaluate_cluster_set(cluster_set)

        result = TargetClusteringResult(
            source_screened_set=screened_set,
            cluster_set=cluster_set,
            records=records,
            metadata={
                **metadata,
                "cluster_metrics": cluster_metrics.to_dict(),
                "method": self.config.method,
                "requested_num_clusters": self.config.num_clusters,
                "actual_num_clusters": actual_num_clusters,
                "num_targets": len(targets),
                "feature_names": list(self.config.feature_names),
                "feature_stats": feature_stats,
            },
        )

        return result

    def write_debug_csv(
        self,
        result: TargetClusteringResult,
        output_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> Path:
        """
        保存目标分群结果，便于检查和后续报告分析。
        """
        path = resolve_path(
            output_path or self.config.debug_csv_path,
            project_root=project_root,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "target_id",
            "cluster_id",
            "x",
            "y",
            "target_type",
            "defense",
            "significance",
            "score",
            "distance_to_cluster_center",
            "feature_vector",
        ]

        score_map = result.source_screened_set.target_scores

        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for record in sorted(
                result.records,
                key=lambda item: (item.cluster_id, item.target.target_id),
            ):
                target = record.target
                target_id = target.target_id
                target_score = score_map.get(target_id)

                writer.writerow(
                    {
                        "target_id": target_id,
                        "cluster_id": record.cluster_id,
                        "x": target.position.x,
                        "y": target.position.y,
                        "target_type": self._target_type_as_int(target),
                        "defense": target.defense,
                        "significance": target.significance,
                        "score": target_score.score if target_score else "",
                        "distance_to_cluster_center": record.distance_to_cluster_center,
                        "feature_vector": record.feature_vector,
                    }
                )

        return path

    def _resolve_num_clusters(
        self,
        num_targets: int,
        requested_num_clusters: int,
    ) -> int:
        """
        解析实际 cluster 数量。

        如果摧毁目标数少于配置的 cluster 数：
        - auto_adjust_num_clusters=True：自动减少 cluster 数；
        - False：直接报错。
        """
        if num_targets <= 0:
            raise TargetClusteringError("num_targets must be positive.")

        if requested_num_clusters <= num_targets:
            return requested_num_clusters

        if self.config.auto_adjust_num_clusters:
            return num_targets

        raise TargetClusteringError(
            "num_targets is smaller than num_clusters: "
            f"num_targets={num_targets}, num_clusters={requested_num_clusters}"
        )

    def _build_feature_matrix(
        self,
        targets: list[Target],
        screened_set: ScreenedTargetSet,
    ) -> tuple[np.ndarray, list[int]]:
        """
        构造分群特征矩阵。

        当前默认使用 x、y。
        后续可以通过 YAML 增加：
        - score
        - defense
        - significance
        - normalized_distance
        """
        features: list[list[float]] = []
        target_ids: list[int] = []

        for target in targets:
            row = [
                self._get_feature_value(
                    target=target,
                    screened_set=screened_set,
                    feature_name=feature_name,
                )
                for feature_name in self.config.feature_names
            ]

            features.append(row)
            target_ids.append(int(target.target_id))

        return np.array(features, dtype=np.float64), target_ids

    def _get_feature_value(
        self,
        target: Target,
        screened_set: ScreenedTargetSet,
        feature_name: str,
    ) -> float:
        """
        读取目标分群特征。
        """
        if feature_name == "x":
            return float(target.position.x)

        if feature_name == "y":
            return float(target.position.y)

        if feature_name == "defense":
            return float(target.defense)

        if feature_name == "significance":
            return float(target.significance)

        if feature_name == "target_type":
            return float(self._target_type_as_int(target))

        target_score = screened_set.target_scores.get(target.target_id)

        if target_score is None:
            raise TargetClusteringError(
                f"TargetScore not found for target_id={target.target_id}"
            )

        if feature_name == "score":
            return float(target_score.score)

        if feature_name in target_score.components:
            return float(target_score.components[feature_name])

        if feature_name in target_score.metadata:
            return float(target_score.metadata[feature_name])

        raise TargetClusteringError(
            f"Unsupported feature_name={feature_name}. "
            "Available basic features: x, y, defense, significance, target_type, score."
        )

    def _normalize_features(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        对分群特征做 min-max 归一化。

        PSO 和 KMeans 都在归一化空间中运行，避免 x/y 数值尺度过大。
        """
        min_values = features.min(axis=0)
        max_values = features.max(axis=0)
        ranges = max_values - min_values

        safe_ranges = np.where(np.abs(ranges) < 1e-12, 1.0, ranges)

        normalized = (features - min_values) / safe_ranges

        return normalized, {
            "min_values": min_values.tolist(),
            "max_values": max_values.tolist(),
            "ranges": safe_ranges.tolist(),
        }

    def _denormalize_features(
        self,
        normalized_features: np.ndarray,
        feature_stats: dict[str, Any],
    ) -> np.ndarray:
        """
        将归一化空间中的中心还原到原始特征空间。
        """
        min_values = np.array(feature_stats["min_values"], dtype=np.float64)
        ranges = np.array(feature_stats["ranges"], dtype=np.float64)

        return normalized_features * ranges + min_values

    def _cluster_by_kmeans(
        self,
        normalized_features: np.ndarray,
        num_clusters: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        KMeans 目标分群。

        这是后续 PSO / PPO 的稳定 baseline。
        """
        num_samples = normalized_features.shape[0]

        rng = np.random.default_rng(self.config.kmeans_random_seed)

        initial_indices = rng.choice(
            num_samples,
            size=num_clusters,
            replace=False,
        )

        centers = normalized_features[initial_indices].copy()
        labels = np.zeros(num_samples, dtype=np.int64)

        previous_error = float("inf")

        for iteration in range(1, self.config.kmeans_max_iter + 1):
            distances = self._pairwise_euclidean(
                normalized_features,
                centers,
            )
            labels = distances.argmin(axis=1)

            new_centers = centers.copy()

            for cluster_id in range(num_clusters):
                cluster_points = normalized_features[labels == cluster_id]

                if len(cluster_points) > 0:
                    new_centers[cluster_id] = cluster_points.mean(axis=0)
                else:
                    replacement_index = int(rng.integers(0, num_samples))
                    new_centers[cluster_id] = normalized_features[replacement_index]

            centers = new_centers

            error = self._clustering_compactness(
                normalized_features=normalized_features,
                labels=labels,
                centers=centers,
            )

            if abs(previous_error - error) < self.config.kmeans_tolerance:
                return labels, centers, {
                    "algorithm": "kmeans",
                    "num_iter": iteration,
                    "objective": error,
                }

            previous_error = error

        return labels, centers, {
            "algorithm": "kmeans",
            "num_iter": self.config.kmeans_max_iter,
            "objective": previous_error,
        }

    def _cluster_by_pso(
        self,
        normalized_features: np.ndarray,
        num_clusters: int,
        targets: list[Target],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        使用 PSO 优化目标群中心。

        每个粒子表示一组 cluster centers：
            shape = (num_clusters, feature_dim)

        目标函数包含：
        - compactness：群内距离尽量小；
        - balance：避免所有目标集中到一个群里。

        这是对源代码 PSO_classify 思想的工程化实现。
        """
        pso = PSOClusterer(
            PSOClustererConfig(
                num_particles=self.config.pso_num_particles,
                max_iter=self.config.pso_max_iter,
                inertia_weight=self.config.pso_inertia_weight,
                cognitive_weight=self.config.pso_cognitive_weight,
                social_weight=self.config.pso_social_weight,
                compactness_weight=self.config.pso_compactness_weight,
                balance_weight=self.config.pso_balance_weight,
                value_weight=self.config.pso_value_weight,
                defense_weight=self.config.pso_defense_weight,
                convergence_threshold=self.config.pso_convergence_threshold,
                random_seed=self.config.pso_random_seed,
                initialize_with_high_value_targets=(
                    self.config.pso_initialize_with_high_value_targets
                ),
            )
        )

        result = pso.cluster(
            features=normalized_features,
            num_clusters=num_clusters,
            target_values=np.asarray(
                [float(target.significance) for target in targets],
                dtype=np.float64,
            ),
            target_defenses=np.asarray(
                [float(target.defense) for target in targets],
                dtype=np.float64,
            ),
        )

        return result.labels, result.centers, {
            "algorithm": "pso",
            "num_iter": result.metadata.get("num_iter", self.config.pso_max_iter),
            "objective": result.best_score,
            "global_best_score": result.best_score,
            "convergence_history": result.convergence_history,
            "pso_metrics": result.metrics,
        }

    def _cluster_by_ppo(
        self,
        normalized_features: np.ndarray,
        num_clusters: int,
        targets: list[Target],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Run PPO dynamic regrouping, falling back to PSO when no policy exists."""
        result = PPOClusterer(
            PPOClustererConfig(
                checkpoint_dir=self.config.ppo_checkpoint_dir,
                fallback_method=self.config.ppo_fallback_method,
                random_seed=self.config.pso_random_seed,
                pso_config=PSOClustererConfig(
                    num_particles=self.config.pso_num_particles,
                    max_iter=self.config.pso_max_iter,
                    inertia_weight=self.config.pso_inertia_weight,
                    cognitive_weight=self.config.pso_cognitive_weight,
                    social_weight=self.config.pso_social_weight,
                    compactness_weight=self.config.pso_compactness_weight,
                    balance_weight=self.config.pso_balance_weight,
                    value_weight=self.config.pso_value_weight,
                    defense_weight=self.config.pso_defense_weight,
                    convergence_threshold=self.config.pso_convergence_threshold,
                    random_seed=self.config.pso_random_seed,
                    initialize_with_high_value_targets=(
                        self.config.pso_initialize_with_high_value_targets
                    ),
                ),
            )
        ).cluster(
            features=normalized_features,
            num_clusters=num_clusters,
            target_values=np.asarray(
                [float(target.significance) for target in targets],
                dtype=np.float64,
            ),
            target_defenses=np.asarray(
                [float(target.defense) for target in targets],
                dtype=np.float64,
            ),
        )
        return result.labels, result.centers, {
            "algorithm": "ppo",
            "method_used": result.method_used,
            "objective": result.metrics.get("mean_distance_to_center"),
            "ppo_metrics": result.metrics,
            "ppo_metadata": result.metadata,
        }

    def _pso_objective(
        self,
        normalized_features: np.ndarray,
        centers: np.ndarray,
    ) -> float:
        """
        PSO 目标函数。

        compactness:
            目标点到所属中心的平均距离。

        balance_penalty:
            不希望某些 cluster 为空或极端不均衡。
        """
        distances = self._pairwise_euclidean(normalized_features, centers)
        labels = distances.argmin(axis=1)

        compactness = self._clustering_compactness(
            normalized_features=normalized_features,
            labels=labels,
            centers=centers,
        )

        counts = np.array(
            [
                np.sum(labels == cluster_id)
                for cluster_id in range(centers.shape[0])
            ],
            dtype=np.float64,
        )

        ideal_count = normalized_features.shape[0] / centers.shape[0]
        balance_penalty = float(
            np.mean((counts - ideal_count) ** 2) / max(ideal_count, 1.0)
        )

        empty_cluster_penalty = float(np.sum(counts == 0)) * 10.0

        return (
            self.config.pso_compactness_weight * compactness
            + self.config.pso_balance_weight * balance_penalty
            + empty_cluster_penalty
        )

    def _recompute_centers_from_labels(
        self,
        normalized_features: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        num_clusters: int,
    ) -> np.ndarray:
        """
        根据标签重新计算 cluster centers。

        如果某个 cluster 为空，则保留 PSO 给出的中心。
        """
        new_centers = centers.copy()

        for cluster_id in range(num_clusters):
            points = normalized_features[labels == cluster_id]

            if len(points) > 0:
                new_centers[cluster_id] = points.mean(axis=0)

        return new_centers

    def _build_cluster_set_and_records(
        self,
        targets: list[Target],
        target_ids: list[int],
        features: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        screened_set: ScreenedTargetSet,
        algorithm_metadata: AlgorithmMetadata,
    ) -> tuple[TargetClusterSet, list[TargetClusteringRecord]]:
        """
        根据 labels 和 centers 构造 TargetClusterSet。
        """
        target_by_id = {
            int(target.target_id): target
            for target in targets
        }

        clusters: list[TargetCluster] = []
        records: list[TargetClusteringRecord] = []

        unique_labels = sorted(set(int(label) for label in labels.tolist()))

        for output_cluster_id, raw_label in enumerate(unique_labels):
            member_indices = [
                index
                for index, label in enumerate(labels)
                if int(label) == raw_label
            ]

            cluster_targets = [
                target_by_id[target_ids[index]]
                for index in member_indices
            ]

            if not cluster_targets:
                continue

            center_position = self._build_center_position(
                cluster_targets=cluster_targets,
                center_features=centers[raw_label],
            )

            defense_sum = sum(float(target.defense) for target in cluster_targets)
            significance_sum = sum(
                float(target.significance) for target in cluster_targets
            )

            compactness = self._cluster_compactness_original_space(
                cluster_targets=cluster_targets,
                center_position=center_position,
            )

            cluster = TargetCluster(
                cluster_id=output_cluster_id,
                targets=cluster_targets,
                center=center_position,
                defense_sum=defense_sum,
                significance_sum=significance_sum,
                compactness=compactness,
                metadata={
                    "raw_label": raw_label,
                    "num_targets": len(cluster_targets),
                    "method": self.config.method,
                },
            )
            cluster.validate()
            clusters.append(cluster)

            for index in member_indices:
                target = target_by_id[target_ids[index]]

                distance_to_center = self._distance_position(
                    target.position,
                    center_position,
                )

                records.append(
                    TargetClusteringRecord(
                        target=target,
                        cluster_id=output_cluster_id,
                        feature_vector=features[index].tolist(),
                        distance_to_cluster_center=distance_to_center,
                        metadata={
                            "raw_label": raw_label,
                        },
                    )
                )

        cluster_set = TargetClusterSet(
            clusters=clusters,
            algorithm_metadata=algorithm_metadata,
            metadata={
                "method": self.config.method,
                "num_clusters": len(clusters),
                "feature_names": list(self.config.feature_names),
            },
        )
        cluster_set.validate()

        return cluster_set, records

    def _build_center_position(
        self,
        cluster_targets: list[Target],
        center_features: np.ndarray,
    ) -> Position:
        """
        构造目标群中心位置。

        如果分群特征包含 x/y，则优先使用 center_features 中的 x/y。
        否则回退到 cluster_targets 的坐标均值。
        """
        feature_to_value = {
            feature_name: float(center_features[index])
            for index, feature_name in enumerate(self.config.feature_names)
        }

        if "x" in feature_to_value and "y" in feature_to_value:
            return Position(
                x=feature_to_value["x"],
                y=feature_to_value["y"],
            )

        return Position(
            x=sum(target.position.x for target in cluster_targets)
            / len(cluster_targets),
            y=sum(target.position.y for target in cluster_targets)
            / len(cluster_targets),
        )

    def _cluster_compactness_original_space(
        self,
        cluster_targets: list[Target],
        center_position: Position,
    ) -> float:
        """计算原始坐标空间中的群内紧凑度。"""
        if not cluster_targets:
            return 0.0

        distances = [
            self._distance_position(target.position, center_position)
            for target in cluster_targets
        ]

        return float(sum(distances) / len(distances))

    @staticmethod
    def _pairwise_euclidean(
        points: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        """计算点到中心的欧氏距离矩阵。"""
        diff = points[:, None, :] - centers[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))

    @staticmethod
    def _clustering_compactness(
        normalized_features: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
    ) -> float:
        """计算归一化空间中的平均群内距离。"""
        total_distance = 0.0

        for index, point in enumerate(normalized_features):
            center = centers[labels[index]]
            total_distance += float(np.linalg.norm(point - center))

        return total_distance / max(len(normalized_features), 1)

    @staticmethod
    def _distance_position(
        position_a: Position,
        position_b: Position,
    ) -> float:
        """计算两个位置之间的欧氏距离。"""
        return math.sqrt(
            (position_a.x - position_b.x) ** 2
            + (position_a.y - position_b.y) ** 2
        )

    @staticmethod
    def _target_type_as_int(target: Target) -> int:
        """尽量把 target_type 转成 int。"""
        target_type = target.target_type

        if hasattr(target_type, "value"):
            return int(target_type.value)

        return int(target_type)


def load_target_clustering_config(
    config: dict[str, Any],
) -> TargetClusteringConfig:
    """从项目总配置中读取目标分群配置。"""
    prefix = "target_clustering"

    clustering_config = TargetClusteringConfig(
        method=str(
            get_config_value(config, f"{prefix}.method", default="kmeans")
        ),
        num_clusters=int(
            get_config_value(config, f"{prefix}.num_clusters", default=3)
        ),
        auto_adjust_num_clusters=bool(
            get_config_value(
                config,
                f"{prefix}.auto_adjust_num_clusters",
                default=True,
            )
        ),
        feature_names=tuple(
            get_config_value(
                config,
                f"{prefix}.feature_names",
                default=["x", "y"],
            )
        ),
        kmeans_max_iter=int(
            get_config_value(
                config,
                f"{prefix}.kmeans.max_iter",
                default=300,
            )
        ),
        kmeans_tolerance=float(
            get_config_value(
                config,
                f"{prefix}.kmeans.tolerance",
                default=0.0004,
            )
        ),
        kmeans_random_seed=int(
            get_config_value(
                config,
                f"{prefix}.kmeans.random_seed",
                default=42,
            )
        ),
        pso_num_particles=int(
            get_config_value(
                config,
                f"{prefix}.pso.num_particles",
                default=30,
            )
        ),
        pso_max_iter=int(
            get_config_value(
                config,
                f"{prefix}.pso.max_iter",
                default=100,
            )
        ),
        pso_inertia_weight=float(
            get_config_value(
                config,
                f"{prefix}.pso.inertia_weight",
                default=0.7,
            )
        ),
        pso_cognitive_weight=float(
            get_config_value(
                config,
                f"{prefix}.pso.cognitive_weight",
                default=1.5,
            )
        ),
        pso_social_weight=float(
            get_config_value(
                config,
                f"{prefix}.pso.social_weight",
                default=1.5,
            )
        ),
        pso_compactness_weight=float(
            get_config_value(
                config,
                f"{prefix}.pso.compactness_weight",
                default=1.0,
            )
        ),
        pso_balance_weight=float(
            get_config_value(
                config,
                f"{prefix}.pso.balance_weight",
                default=0.1,
            )
        ),
        pso_value_weight=float(
            get_config_value(
                config,
                f"{prefix}.pso.value_weight",
                default=get_config_value(config, "pso.weights.value", default=0.2),
            )
        ),
        pso_defense_weight=float(
            get_config_value(
                config,
                f"{prefix}.pso.defense_weight",
                default=get_config_value(config, "pso.weights.defense", default=0.2),
            )
        ),
        pso_convergence_threshold=float(
            get_config_value(
                config,
                f"{prefix}.pso.convergence_threshold",
                default=get_config_value(
                    config,
                    "pso.convergence_threshold",
                    default=1e-6,
                ),
            )
        ),
        pso_initialize_with_high_value_targets=bool(
            get_config_value(
                config,
                f"{prefix}.pso.initialize_with_high_value_targets",
                default=True,
            )
        ),
        pso_random_seed=int(
            get_config_value(
                config,
                f"{prefix}.pso.random_seed",
                default=42,
            )
        ),
        ppo_checkpoint_dir=str(
            get_config_value(
                config,
                "ppo.checkpoint_dir",
                default="checkpoints/ppo_clusterer",
            )
        ),
        ppo_fallback_method=str(
            get_config_value(
                config,
                "ppo.fallback_method",
                default="pso",
            )
        ),
        debug_csv_path=str(
            get_config_value(
                config,
                f"{prefix}.output.debug_csv_path",
                default="outputs/intermediate/target_clustering.csv",
            )
        ),
    )

    clustering_config.validate()
    return clustering_config
