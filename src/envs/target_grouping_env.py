"""envs 数据模块中的目标grouping环境实现。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TargetGroupingEnvConfig:
    """Configuration for the dynamic target regrouping PPO environment."""

    # num_clusters: num目标簇集合。
    num_clusters: int = 3
    # max_steps: 最大值步数。
    max_steps: int = 30
    # max_targets_per_cluster: 最大值目标集合per目标簇。
    max_targets_per_cluster: int = 32
    # illegal_action_penalty: illegal动作penalty。
    illegal_action_penalty: float = -1.0
    # count_balance_weight: countbalance权重。
    count_balance_weight: float = 1.0
    # compactness_weight: compactness权重。
    compactness_weight: float = 1.0
    # workload_balance_weight: workloadbalance权重。
    workload_balance_weight: float = 1.0
    # dynamic_event_bonus: 动态事件bonus。
    dynamic_event_bonus: float = 0.05

    def validate(self) -> None:
        """校验当前对象或输入配置的合法性。

        参数：
            无显式业务参数。

        返回：
            无返回值；通过状态变更、文件输出或日志记录体现执行结果。
        """
        if self.num_clusters <= 0:
            raise ValueError("num_clusters must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.max_targets_per_cluster <= 0:
            raise ValueError("max_targets_per_cluster must be positive.")


class TargetGroupingEnv:
    """TargetGroupingEnv 类，封装目标grouping环境相关的数据结构与业务行为。

    属性：
        features: features 数据。
        initial_labels: initiallabels。
        values: values 数据。
        defenses: defenses 数据。
        config: 配置。
        labels: labels 数据。
        current_step: 当前步数。
    """

    observation_features_per_cluster = 8

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        config: TargetGroupingEnvConfig,
        target_values: np.ndarray | None = None,
        target_defenses: np.ndarray | None = None,
    ) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            features: features 参数，类型为 np.ndarray。
            labels: labels 参数，类型为 np.ndarray。
            config: 配置对象，类型为 TargetGroupingEnvConfig。
            target_values: target_values 参数，类型为 np.ndarray | None。
            target_defenses: target_defenses 参数，类型为 np.ndarray | None。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        config.validate()
        # features: features 数据。
        self.features = np.asarray(features, dtype=np.float64)
        if self.features.ndim != 2 or self.features.shape[1] < 2:
            raise ValueError("features must be a 2D array with at least x/y columns.")

        # initial_labels: initiallabels。
        self.initial_labels = np.asarray(labels, dtype=np.int64)
        if self.initial_labels.ndim != 1 or len(self.initial_labels) != len(self.features):
            raise ValueError("labels must be a 1D array with one label per target.")

        # values: values 数据。
        self.values = (
            np.asarray(target_values, dtype=np.float64)
            if target_values is not None
            else np.ones(len(self.features), dtype=np.float64)
        )
        # defenses: defenses 数据。
        self.defenses = (
            np.asarray(target_defenses, dtype=np.float64)
            if target_defenses is not None
            else np.ones(len(self.features), dtype=np.float64)
        )
        if len(self.values) != len(self.features) or len(self.defenses) != len(self.features):
            raise ValueError("target_values and target_defenses must match feature length.")

        # config: 配置。
        self.config = config
        # labels: labels 数据。
        self.labels = self.initial_labels.copy()
        # current_step: 当前步数。
        self.current_step = 0

    @property
    def action_dim(self) -> int:
        """Flattened discrete action dimension."""
        return (
            self.config.num_clusters
            * self.config.max_targets_per_cluster
            * self.config.num_clusters
        )

    @property
    def observation_dim(self) -> int:
        """Flattened observation dimension."""
        return self.config.num_clusters * self.observation_features_per_cluster

    def reset(self) -> np.ndarray:
        """重置对象状态，为新的回合或流程做准备。

        参数：
            无显式业务参数。

        返回：
            np.ndarray，表示该函数计算或构建得到的结果。
        """
        self.labels = self.initial_labels.copy()
        self.current_step = 0
        return self._observation()

    def encode_action(
        self,
        source_cluster: int,
        source_slot: int,
        destination_cluster: int,
    ) -> int:
        """Encode a triple action as a flattened discrete action id."""
        return int(
            source_cluster * self.config.max_targets_per_cluster * self.config.num_clusters
            + source_slot * self.config.num_clusters
            + destination_cluster
        )

    def decode_action(self, action_id: int) -> tuple[int, int, int]:
        """Decode a flattened discrete action id to a triple action."""
        if action_id < 0 or action_id >= self.action_dim:
            raise ValueError(f"action_id out of range: {action_id}")
        destination_cluster = action_id % self.config.num_clusters
        source_slot = (action_id // self.config.num_clusters) % self.config.max_targets_per_cluster
        source_cluster = action_id // (
            self.config.max_targets_per_cluster * self.config.num_clusters
        )
        return int(source_cluster), int(source_slot), int(destination_cluster)

    def action_mask(self) -> np.ndarray:
        """Return a binary mask where 1 means a legal regrouping action."""
        mask = np.zeros(self.action_dim, dtype=np.float32)
        for source_cluster in range(self.config.num_clusters):
            source_count = int(np.sum(self.labels == source_cluster))
            legal_slots = min(source_count, self.config.max_targets_per_cluster)
            for source_slot in range(legal_slots):
                for destination_cluster in range(self.config.num_clusters):
                    if source_cluster == destination_cluster:
                        continue
                    action_id = self.encode_action(
                        source_cluster=source_cluster,
                        source_slot=source_slot,
                        destination_cluster=destination_cluster,
                    )
                    mask[action_id] = 1.0
        return mask

    def step_index(self, action_id: int):
        """Step with a flattened discrete action id."""
        return self.step(self.decode_action(int(action_id)))

    def step(self, action: tuple[int, int, int]):
        """推进环境或仿真流程的一个时间步。

        参数：
            action: 动作，类型为 tuple[int, int, int]。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        source_cluster, source_slot, destination_cluster = action
        before_score = self._score()
        info: dict[str, Any] = {"legal_action": False}

        if not self._is_legal_action(source_cluster, source_slot, destination_cluster):
            reward = float(self.config.illegal_action_penalty)
            info["reason"] = "illegal_action"
        else:
            source_members = np.where(self.labels == source_cluster)[0]
            target_index = int(source_members[source_slot])
            self.labels[target_index] = destination_cluster
            after_score = self._score()
            reward = float(after_score - before_score + self.config.dynamic_event_bonus)
            info.update(
                {
                    "legal_action": True,
                    "moved_target_index": target_index,
                    "score_before": before_score,
                    "score_after": after_score,
                }
            )

        self.current_step += 1
        done = bool(self.current_step >= self.config.max_steps or self.action_mask().sum() == 0)
        info.update(self.get_metrics())
        return self._observation(), reward, done, info

    def get_metrics(self) -> dict[str, float]:
        """Return compactness and balance metrics for logging/evaluation."""
        counts = self._cluster_counts()
        mean_distance = self._mean_intra_cluster_distance()
        workloads = self._cluster_workloads()
        count_balance = float(1.0 / (1.0 + counts.std()))
        compactness = float(1.0 / (1.0 + mean_distance))
        workload_balance = float(1.0 / (1.0 + workloads.std()))
        return {
            "count_balance": count_balance,
            "compactness": compactness,
            "workload_balance": workload_balance,
            "score": self._score(),
            "count_std": float(counts.std()),
            "workload_std": float(workloads.std()),
            "mean_intra_cluster_distance": float(mean_distance),
        }

    def _is_legal_action(
        self,
        source_cluster: int,
        source_slot: int,
        destination_cluster: int,
    ) -> bool:
        """处理islegal动作相关业务逻辑。

        参数：
            source_cluster: source目标簇，类型为 int。
            source_slot: sourceslot，类型为 int。
            destination_cluster: destination目标簇，类型为 int。

        返回：
            bool，表示该函数计算或构建得到的结果。
        """
        if source_cluster == destination_cluster:
            return False
        if source_cluster < 0 or destination_cluster < 0:
            return False
        if source_cluster >= self.config.num_clusters:
            return False
        if destination_cluster >= self.config.num_clusters:
            return False
        if source_slot < 0 or source_slot >= self.config.max_targets_per_cluster:
            return False
        source_count = int(np.sum(self.labels == source_cluster))
        return source_slot < source_count

    def _observation(self) -> np.ndarray:
        """处理观测向量相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            np.ndarray，表示该函数计算或构建得到的结果。
        """
        rows = []
        coordinates = self.features[:, :2]
        for cluster_id in range(self.config.num_clusters):
            member_indices = np.where(self.labels == cluster_id)[0]
            if len(member_indices) == 0:
                rows.append([0.0] * self.observation_features_per_cluster)
                continue

            members = coordinates[member_indices]
            center = members.mean(axis=0)
            distances = np.linalg.norm(members - center, axis=1)
            value_sum = float(self.values[member_indices].sum())
            defense_sum = float(self.defenses[member_indices].sum())
            rows.append(
                [
                    float(len(member_indices)),
                    float(center[0]),
                    float(center[1]),
                    float(members.std()),
                    float(distances.mean()) if len(distances) else 0.0,
                    value_sum,
                    defense_sum,
                    value_sum / max(defense_sum, 1e-6),
                ]
            )
        return np.asarray(rows, dtype=np.float32).reshape(-1)

    def _score(self) -> float:
        """处理评分相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            float，表示该函数计算或构建得到的结果。
        """
        metrics = self._score_components()
        return float(
            self.config.count_balance_weight * metrics["count_balance"]
            + self.config.compactness_weight * metrics["compactness"]
            + self.config.workload_balance_weight * metrics["workload_balance"]
        )

    def _score_components(self) -> dict[str, float]:
        """处理评分components相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            dict[str, float]，表示该函数计算或构建得到的结果。
        """
        counts = self._cluster_counts()
        workloads = self._cluster_workloads()
        mean_distance = self._mean_intra_cluster_distance()
        return {
            "count_balance": float(1.0 / (1.0 + counts.std())),
            "compactness": float(1.0 / (1.0 + mean_distance)),
            "workload_balance": float(1.0 / (1.0 + workloads.std())),
        }

    def _cluster_counts(self) -> np.ndarray:
        """处理目标簇counts相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            np.ndarray，表示该函数计算或构建得到的结果。
        """
        return np.asarray(
            [np.sum(self.labels == cluster_id) for cluster_id in range(self.config.num_clusters)],
            dtype=np.float64,
        )

    def _cluster_workloads(self) -> np.ndarray:
        """处理目标簇workloads相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            np.ndarray，表示该函数计算或构建得到的结果。
        """
        workloads = []
        for cluster_id in range(self.config.num_clusters):
            member_indices = np.where(self.labels == cluster_id)[0]
            if len(member_indices) == 0:
                workloads.append(0.0)
            else:
                workloads.append(
                    float(self.values[member_indices].sum() + self.defenses[member_indices].sum())
                )
        return np.asarray(workloads, dtype=np.float64)

    def _mean_intra_cluster_distance(self) -> float:
        """处理均值intra目标簇distance相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            float，表示该函数计算或构建得到的结果。
        """
        coordinates = self.features[:, :2]
        distances: list[float] = []
        for cluster_id in range(self.config.num_clusters):
            member_indices = np.where(self.labels == cluster_id)[0]
            if len(member_indices) <= 1:
                continue
            members = coordinates[member_indices]
            center = members.mean(axis=0)
            distances.append(float(np.linalg.norm(members - center, axis=1).mean()))
        return float(np.mean(distances)) if distances else 0.0
