"""PPO environment for dynamic target regrouping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TargetGroupingEnvConfig:
    """Configuration for the dynamic target regrouping environment."""

    num_clusters: int = 3
    max_steps: int = 30
    max_targets_per_cluster: int = 64
    illegal_action_penalty: float = -1.0
    count_balance_weight: float = 1.0
    compactness_weight: float = 1.0
    workload_balance_weight: float = 1.0
    empty_cluster_penalty: float = -1.0
    move_penalty: float = -0.01
    improvement_bonus: float = 0.05
    patience: int = 8
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0
    use_reward_clipping: bool = True
    use_score_delta_reward: bool = True

    # Backward-compatible alias used by the older toy trainer config.
    dynamic_event_bonus: float | None = None

    def validate(self) -> None:
        if self.num_clusters <= 0:
            raise ValueError("num_clusters must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.max_targets_per_cluster <= 0:
            raise ValueError("max_targets_per_cluster must be positive.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if self.reward_clip_min > self.reward_clip_max:
            raise ValueError("reward_clip_min must be <= reward_clip_max.")


class TargetGroupingEnv:
    """
    Environment for adjusting target clusters after dynamic battlefield events.

    Action:
        (source_cluster, source_slot, destination_cluster)
    """

    observation_features_per_cluster = 9

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        config: TargetGroupingEnvConfig,
        target_values: np.ndarray | None = None,
        target_defenses: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        config.validate()
        self.features = np.asarray(features, dtype=np.float64)
        if self.features.ndim != 2 or self.features.shape[1] < 2:
            raise ValueError("features must be a 2D array with at least x/y columns.")

        self.initial_labels = np.asarray(labels, dtype=np.int64)
        if self.initial_labels.ndim != 1 or len(self.initial_labels) != len(self.features):
            raise ValueError("labels must be a 1D array with one label per target.")
        if len(self.features) < config.num_clusters:
            raise ValueError("num targets must be >= num_clusters.")

        self.values = (
            np.asarray(target_values, dtype=np.float64)
            if target_values is not None
            else np.ones(len(self.features), dtype=np.float64)
        )
        self.defenses = (
            np.asarray(target_defenses, dtype=np.float64)
            if target_defenses is not None
            else np.ones(len(self.features), dtype=np.float64)
        )
        if len(self.values) != len(self.features) or len(self.defenses) != len(self.features):
            raise ValueError("target_values and target_defenses must match feature length.")

        self.config = config
        self.metadata = metadata or {}
        self.labels = self.initial_labels.copy()
        self.current_step = 0
        self.best_score = 0.0
        self.no_improvement_steps = 0

    @property
    def action_dim(self) -> int:
        return (
            self.config.num_clusters
            * self.config.max_targets_per_cluster
            * self.config.num_clusters
        )

    @property
    def observation_dim(self) -> int:
        return self.config.num_clusters * self.observation_features_per_cluster

    def reset(self) -> np.ndarray:
        self.labels = self.initial_labels.copy()
        self.current_step = 0
        self.best_score = self._score()
        self.no_improvement_steps = 0
        return self._observation()

    def encode_action(
        self,
        source_cluster: int,
        source_slot: int,
        destination_cluster: int,
    ) -> int:
        return int(
            source_cluster * self.config.max_targets_per_cluster * self.config.num_clusters
            + source_slot * self.config.num_clusters
            + destination_cluster
        )

    def decode_action(self, action_id: int) -> tuple[int, int, int]:
        if action_id < 0 or action_id >= self.action_dim:
            raise ValueError(f"action_id out of range: {action_id}")
        destination_cluster = action_id % self.config.num_clusters
        source_slot = (action_id // self.config.num_clusters) % self.config.max_targets_per_cluster
        source_cluster = action_id // (
            self.config.max_targets_per_cluster * self.config.num_clusters
        )
        return int(source_cluster), int(source_slot), int(destination_cluster)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=np.float32)
        for source_cluster in range(self.config.num_clusters):
            source_count = int(np.sum(self.labels == source_cluster))
            legal_slots = min(source_count, self.config.max_targets_per_cluster)
            for source_slot in range(legal_slots):
                for destination_cluster in range(self.config.num_clusters):
                    if source_cluster == destination_cluster:
                        continue
                    mask[
                        self.encode_action(
                            source_cluster=source_cluster,
                            source_slot=source_slot,
                            destination_cluster=destination_cluster,
                        )
                    ] = 1.0
        return mask

    def get_action_mask(self) -> np.ndarray:
        return self.action_mask()

    def step_index(self, action_id: int):
        try:
            action = self.decode_action(int(action_id))
        except ValueError:
            return self._illegal_step("action_id_out_of_range")
        return self.step(action)

    def step(self, action: tuple[int, int, int]):
        source_cluster, source_slot, destination_cluster = action
        before_score = self._score()
        info: dict[str, Any] = {
            "legal_action": False,
            "score_before": before_score,
        }

        if not self._is_legal_action(source_cluster, source_slot, destination_cluster):
            self.current_step += 1
            raw_reward = float(self.config.illegal_action_penalty)
            reward = self._clip_reward(raw_reward)
            info["reason"] = "illegal_action"
            info["raw_reward"] = raw_reward
            info["clipped_reward"] = reward
            done = self._is_done()
            info.update(self.get_metrics())
            return self._observation(), reward, done, info

        source_members = np.where(self.labels == source_cluster)[0]
        target_index = int(source_members[source_slot])
        self.labels[target_index] = destination_cluster

        after_score = self._score()
        improvement = after_score - before_score
        improved_best = after_score > self.best_score + 1e-12
        if improved_best:
            self.best_score = after_score
            self.no_improvement_steps = 0
        else:
            self.no_improvement_steps += 1

        base_reward = improvement if self.config.use_score_delta_reward else after_score
        raw_reward = (
            base_reward
            + (self.config.improvement_bonus if improvement > 0 else 0.0)
            + self.config.move_penalty
        )
        reward = self._clip_reward(raw_reward)

        self.current_step += 1
        done = self._is_done()
        info.update(
            {
                "legal_action": True,
                "moved_target_index": target_index,
                "score_after": after_score,
                "improvement": improvement,
                "raw_reward": raw_reward,
                "clipped_reward": reward,
                "improved_best": improved_best,
            }
        )
        info.update(self.get_metrics())
        return self._observation(), float(reward), done, info

    def get_current_labels(self) -> np.ndarray:
        return self.labels.copy()

    def get_metrics(self) -> dict[str, float]:
        components = self._score_components()
        counts = self._cluster_counts()
        workloads = self._cluster_workloads()
        final_score = self._score()
        return {
            "final_score": final_score,
            "score": final_score,
            "count_balance": components["count_balance_score"],
            "count_balance_score": components["count_balance_score"],
            "compactness": components["compactness_score"],
            "compactness_score": components["compactness_score"],
            "workload_balance": components["workload_balance_score"],
            "workload_balance_score": components["workload_balance_score"],
            "count_std": float(counts.std()),
            "workload_std": float(workloads.std()),
            "mean_intra_cluster_distance": components["mean_intra_cluster_distance"],
            "empty_cluster_count": float(np.sum(counts == 0)),
        }

    def _illegal_step(self, reason: str):
        self.current_step += 1
        raw_reward = float(self.config.illegal_action_penalty)
        reward = self._clip_reward(raw_reward)
        done = self._is_done()
        info: dict[str, Any] = {
            "legal_action": False,
            "reason": reason,
            "raw_reward": raw_reward,
            "clipped_reward": reward,
        }
        info.update(self.get_metrics())
        return self._observation(), reward, done, info

    def _clip_reward(self, reward: float) -> float:
        if not self.config.use_reward_clipping:
            return float(reward)
        return float(
            np.clip(
                reward,
                self.config.reward_clip_min,
                self.config.reward_clip_max,
            )
        )

    def _is_done(self) -> bool:
        return bool(
            self.current_step >= self.config.max_steps
            or self.action_mask().sum() == 0
            or self.no_improvement_steps >= self.config.patience
        )

    def _is_legal_action(
        self,
        source_cluster: int,
        source_slot: int,
        destination_cluster: int,
    ) -> bool:
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
        rows: list[list[float]] = []
        coordinates = self.features[:, :2]
        total_targets = max(len(self.features), 1)
        total_workload = max(float((self.values + self.defenses).sum()), 1e-8)

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
            workload = value_sum + defense_sum
            coordinate_std = float(np.mean(np.std(members, axis=0)))

            rows.append(
                [
                    float(len(member_indices)) / total_targets,
                    float(center[0]),
                    float(center[1]),
                    float(distances.mean()) if len(distances) else 0.0,
                    coordinate_std,
                    value_sum,
                    defense_sum,
                    workload,
                    workload / total_workload,
                ]
            )
        return np.asarray(rows, dtype=np.float32).reshape(-1)

    def _score(self) -> float:
        metrics = self._score_components()
        return float(
            self.config.count_balance_weight * metrics["count_balance_score"]
            + self.config.compactness_weight * metrics["compactness_score"]
            + self.config.workload_balance_weight * metrics["workload_balance_score"]
            + self.config.empty_cluster_penalty * metrics["empty_cluster_count"]
        )

    def _score_components(self) -> dict[str, float]:
        counts = self._cluster_counts()
        workloads = self._cluster_workloads()
        mean_distance = self._mean_intra_cluster_distance()

        count_mean = max(float(counts.mean()), 1e-8)
        workload_mean = max(float(workloads.mean()), 1e-8)
        return {
            "count_balance_score": float(1.0 / (1.0 + float(counts.std()) / count_mean)),
            "compactness_score": float(1.0 / (1.0 + mean_distance)),
            "workload_balance_score": float(
                1.0 / (1.0 + float(workloads.std()) / workload_mean)
            ),
            "empty_cluster_count": float(np.sum(counts == 0)),
            "mean_intra_cluster_distance": mean_distance,
        }

    def _cluster_counts(self) -> np.ndarray:
        return np.asarray(
            [
                np.sum(self.labels == cluster_id)
                for cluster_id in range(self.config.num_clusters)
            ],
            dtype=np.float64,
        )

    def _cluster_workloads(self) -> np.ndarray:
        workloads: list[float] = []
        for cluster_id in range(self.config.num_clusters):
            member_indices = np.where(self.labels == cluster_id)[0]
            if len(member_indices) == 0:
                workloads.append(0.0)
            else:
                workloads.append(
                    float(
                        self.values[member_indices].sum()
                        + self.defenses[member_indices].sum()
                    )
                )
        return np.asarray(workloads, dtype=np.float64)

    def _mean_intra_cluster_distance(self) -> float:
        coordinates = self.features[:, :2]
        distances: list[float] = []
        for cluster_id in range(self.config.num_clusters):
            member_indices = np.where(self.labels == cluster_id)[0]
            if len(member_indices) <= 1:
                continue
            members = coordinates[member_indices]
            center = members.mean(axis=0)
            distances.extend(np.linalg.norm(members - center, axis=1).tolist())
        return float(np.mean(distances)) if distances else 0.0
