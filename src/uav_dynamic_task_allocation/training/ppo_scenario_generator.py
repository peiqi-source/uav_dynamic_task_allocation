"""Scenario generator for real-data PPO target regrouping episodes."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from uav_dynamic_task_allocation.allocation.pso_clusterer import (
    PSOClusterer,
    PSOClustererConfig,
)
from uav_dynamic_task_allocation.core.entities import Target


@dataclass
class PPOGroupingScenario:
    """One PPO training episode scenario."""

    scenario_id: int
    event_type: str
    targets: list[Target]
    features: np.ndarray
    initial_labels: np.ndarray
    target_values: np.ndarray
    target_defenses: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PPOGroupingScenarioGeneratorConfig:
    """Configuration for dynamic PPO episode generation."""

    feature_names: tuple[str, ...] = ("x", "y")
    num_clusters: int = 3
    random_seed: int = 42
    sampling_mode: str = "random"
    probabilities: dict[str, float] | None = None
    curriculum: dict[str, Any] | None = None
    target_removed: dict[str, int] | None = None
    target_added: dict[str, int] | None = None
    target_value_changed: dict[str, Any] | None = None
    mixed: dict[str, int] | None = None
    pso_config: PSOClustererConfig = field(default_factory=PSOClustererConfig)


class PPOGroupingScenarioGenerator:
    """
    Generate dynamic regrouping episodes from real screened destroy targets.

    PPO uses these scenarios as interaction sources instead of supervised labels.
    """

    SUPPORTED_EVENTS = {
        "none",
        "target_removed",
        "target_added",
        "target_value_changed",
        "mixed",
    }

    def __init__(
        self,
        destroy_targets: list[Target],
        non_destroy_targets: list[Target],
        target_scores: dict[int, Any],
        config: PPOGroupingScenarioGeneratorConfig,
    ) -> None:
        if config.num_clusters <= 0:
            raise ValueError("num_clusters must be positive.")
        if len(destroy_targets) < config.num_clusters:
            raise ValueError("destroy_targets must contain at least num_clusters targets.")
        self.destroy_targets = list(destroy_targets)
        self.non_destroy_targets = list(non_destroy_targets)
        self.target_scores = target_scores
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.probabilities = self._normalize_probabilities(config.probabilities)
        self.curriculum = config.curriculum or {}
        if config.sampling_mode not in {"random", "curriculum"}:
            raise ValueError("sampling_mode must be 'random' or 'curriculum'.")

    def generate_episode_scenario(self, episode_id: int) -> PPOGroupingScenario:
        """Generate one reproducible dynamic episode scenario."""
        for _ in range(30):
            event_type = self._sample_event_type(episode_id)
            targets, metadata = self._apply_event(event_type)
            if len(targets) >= self.config.num_clusters:
                return self._build_scenario(
                    episode_id=episode_id,
                    event_type=event_type,
                    targets=targets,
                    metadata=metadata,
                )

        targets, metadata = self._apply_event("none")
        return self._build_scenario(
            episode_id=episode_id,
            event_type="none",
            targets=targets,
            metadata=metadata,
        )

    def _build_scenario(
        self,
        episode_id: int,
        event_type: str,
        targets: list[Target],
        metadata: dict[str, Any],
    ) -> PPOGroupingScenario:
        raw_features = self._build_feature_matrix(targets)
        features = self._normalize_features(raw_features)
        target_values = np.asarray(
            [float(target.significance) for target in targets],
            dtype=np.float64,
        )
        target_defenses = np.asarray(
            [float(target.defense) for target in targets],
            dtype=np.float64,
        )

        pso_result = PSOClusterer(self.config.pso_config).cluster(
            features=features,
            num_clusters=self.config.num_clusters,
            target_values=target_values,
            target_defenses=target_defenses,
        )

        scenario_metadata = {
            "scenario_id": episode_id,
            "event_type": event_type,
            "num_targets": len(targets),
            "num_added": metadata.get("num_added", 0),
            "num_removed": metadata.get("num_removed", 0),
            "num_changed": metadata.get("num_changed", 0),
            "feature_names": list(self.config.feature_names),
            "initial_clustering_method": "pso",
            "sampling_mode": self.config.sampling_mode,
        }
        return PPOGroupingScenario(
            scenario_id=episode_id,
            event_type=event_type,
            targets=targets,
            features=features,
            initial_labels=pso_result.labels,
            target_values=target_values,
            target_defenses=target_defenses,
            metadata=scenario_metadata,
        )

    def _apply_event(self, event_type: str) -> tuple[list[Target], dict[str, int]]:
        targets = list(self.destroy_targets)
        metadata = {"num_added": 0, "num_removed": 0, "num_changed": 0}

        if event_type == "none":
            return targets, metadata

        if event_type in {"target_removed", "mixed"}:
            max_remove = self._event_int(event_type, "max_remove", 5 if event_type == "target_removed" else 3)
            min_remove = self._event_int(event_type, "min_remove", 1)
            removable = max(0, len(targets) - self.config.num_clusters)
            remove_count = min(removable, self._randint(min_remove, max_remove))
            if remove_count > 0:
                remove_indices = set(
                    int(index)
                    for index in self.rng.choice(
                        len(targets),
                        size=remove_count,
                        replace=False,
                    )
                )
                targets = [
                    target for index, target in enumerate(targets)
                    if index not in remove_indices
                ]
                metadata["num_removed"] = remove_count

        if event_type in {"target_added", "mixed"} and self.non_destroy_targets:
            max_add = self._event_int(event_type, "max_add", 5 if event_type == "target_added" else 3)
            min_add = self._event_int(event_type, "min_add", 1)
            add_count = min(len(self.non_destroy_targets), self._randint(min_add, max_add))
            if add_count > 0:
                add_indices = self.rng.choice(
                    len(self.non_destroy_targets),
                    size=add_count,
                    replace=False,
                )
                targets.extend(self.non_destroy_targets[int(index)] for index in add_indices)
                metadata["num_added"] = add_count

        if event_type in {"target_value_changed", "mixed"}:
            max_change = self._event_int(
                event_type,
                "max_change",
                8 if event_type == "target_value_changed" else 5,
            )
            min_change = self._event_int(event_type, "min_change", 1)
            change_count = min(len(targets), self._randint(min_change, max_change))
            if change_count > 0:
                value_range = self._event_range(
                    "target_value_changed",
                    "value_scale_range",
                    (0.8, 1.3),
                )
                defense_range = self._event_range(
                    "target_value_changed",
                    "defense_scale_range",
                    (0.8, 1.3),
                )
                change_indices = set(
                    int(index)
                    for index in self.rng.choice(
                        len(targets),
                        size=change_count,
                        replace=False,
                    )
                )
                changed_targets: list[Target] = []
                for index, target in enumerate(targets):
                    if index not in change_indices:
                        changed_targets.append(target)
                        continue
                    value_scale = float(self.rng.uniform(value_range[0], value_range[1]))
                    defense_scale = float(self.rng.uniform(defense_range[0], defense_range[1]))
                    changed_targets.append(
                        replace(
                            target,
                            significance=max(0.0, float(target.significance) * value_scale),
                            defense=max(0.0, float(target.defense) * defense_scale),
                        )
                    )
                targets = changed_targets
                metadata["num_changed"] = change_count

        return targets, metadata

    def _build_feature_matrix(self, targets: list[Target]) -> np.ndarray:
        return np.asarray(
            [
                [
                    self._feature_value(target=target, feature_name=feature_name)
                    for feature_name in self.config.feature_names
                ]
                for target in targets
            ],
            dtype=np.float64,
        )

    def _feature_value(self, target: Target, feature_name: str) -> float:
        if feature_name == "x":
            return float(target.position.x)
        if feature_name == "y":
            return float(target.position.y)
        if feature_name == "defense":
            return float(target.defense)
        if feature_name == "significance":
            return float(target.significance)
        if feature_name == "target_type":
            return float(target.target_type.value if hasattr(target.target_type, "value") else target.target_type)

        target_score = self.target_scores.get(int(target.target_id))
        if target_score is None:
            raise ValueError(f"TargetScore not found for target_id={target.target_id}")
        if feature_name == "score":
            return float(target_score.score)
        if feature_name in target_score.components:
            return float(target_score.components[feature_name])
        if feature_name in target_score.metadata:
            return float(target_score.metadata[feature_name])
        raise ValueError(f"Unsupported feature_name={feature_name}")

    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        min_values = features.min(axis=0)
        max_values = features.max(axis=0)
        ranges = max_values - min_values
        safe_ranges = np.where(np.abs(ranges) < 1e-12, 1.0, ranges)
        return (features - min_values) / safe_ranges

    def _sample_event_type(self, episode_id: int) -> str:
        probabilities = self._episode_probabilities(episode_id)
        names = list(probabilities.keys())
        probs = np.asarray([probabilities[name] for name in names], dtype=np.float64)
        return str(self.rng.choice(names, p=probs))

    def _episode_probabilities(self, episode_id: int) -> dict[str, float]:
        if self.config.sampling_mode != "curriculum":
            return self.probabilities

        for _, stage_config in sorted(self.curriculum.items()):
            episode_range = stage_config.get("episode_range", [0, 999999])
            start = int(episode_range[0])
            end = int(episode_range[1])
            if start <= episode_id < end:
                return self._normalize_probabilities(
                    stage_config.get("probabilities", self.probabilities)
                )

        return self.probabilities

    def _normalize_probabilities(
        self,
        probabilities: dict[str, float] | None,
    ) -> dict[str, float]:
        values = probabilities or {
            "none": 0.25,
            "target_removed": 0.25,
            "target_added": 0.25,
            "target_value_changed": 0.15,
            "mixed": 0.10,
        }
        unknown = set(values) - self.SUPPORTED_EVENTS
        if unknown:
            raise ValueError(f"Unsupported PPO scenario events: {sorted(unknown)}")
        total = float(sum(max(float(value), 0.0) for value in values.values()))
        if total <= 0:
            raise ValueError("scenario event probabilities must sum to a positive value.")
        return {
            event: max(float(probability), 0.0) / total
            for event, probability in values.items()
        }

    def _event_int(self, event_type: str, key: str, default: int) -> int:
        event_config = getattr(self.config, event_type, None) or {}
        fallback_config = self.config.target_value_changed or {}
        value = event_config.get(key, fallback_config.get(key, default))
        return int(value)

    def _event_range(
        self,
        event_type: str,
        key: str,
        default: tuple[float, float],
    ) -> tuple[float, float]:
        event_config = getattr(self.config, event_type, None) or {}
        values = event_config.get(key, default)
        return float(values[0]), float(values[1])

    def _randint(self, min_value: int, max_value: int) -> int:
        low = max(0, int(min_value))
        high = max(low, int(max_value))
        return int(self.rng.integers(low, high + 1))
