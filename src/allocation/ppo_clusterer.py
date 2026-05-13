from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from uav_dynamic_task_allocation.algorithms.rl.ppo.checkpoint import load_ppo_metadata

from uav_dynamic_task_allocation.allocation.pso_clusterer import (
    PSOClusterer,
    PSOClustererConfig,
)


class PPOClustererError(Exception):
    """Raised when PPO target regrouping cannot be executed."""


@dataclass(frozen=True)
class PPOClustererConfig:
    """Configuration for PPO-based dynamic target regrouping."""

    checkpoint_dir: str = "checkpoints/ppo_clusterer"
    fallback_method: str = "pso"
    random_seed: int = 42
    pso_config: PSOClustererConfig = field(default_factory=PSOClustererConfig)


@dataclass
class PPOClustererResult:
    """Dynamic target regrouping result."""

    labels: np.ndarray
    centers: np.ndarray
    method_used: str
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PPOClusterer:
    """
    PPO dynamic regrouping adapter.

    This class provides the formal engineering entry for dynamic regrouping.
    If a trained PPO checkpoint is not present, it falls back to PSO so the
    event-driven mission pipeline remains runnable and reproducible.
    """

    def __init__(self, config: PPOClustererConfig) -> None:
        self.config = config

    def cluster(
        self,
        features: np.ndarray,
        num_clusters: int,
        target_values: np.ndarray | None = None,
        target_defenses: np.ndarray | None = None,
    ) -> PPOClustererResult:
        metadata = load_ppo_metadata(self.config.checkpoint_dir)

        if self.config.fallback_method != "pso":
            raise PPOClustererError(
                f"Unsupported PPO fallback_method={self.config.fallback_method}"
            )

        pso_result = PSOClusterer(self.config.pso_config).cluster(
            features=features,
            num_clusters=num_clusters,
            target_values=target_values,
            target_defenses=target_defenses,
        )
        return PPOClustererResult(
            labels=pso_result.labels,
            centers=pso_result.centers,
            method_used="ppo_fallback_pso",
            metrics=pso_result.metrics,
            metadata={
                **metadata,
                "fallback_method": "pso",
                "reason": self._fallback_reason(metadata),
            },
        )

    def _checkpoint_path(self) -> Path:
        return Path(self.config.checkpoint_dir) / "ppo_clusterer_checkpoint.json"

    @staticmethod
    def _fallback_reason(metadata: dict[str, Any]) -> str:
        status = metadata.get("status")
        if status == "no_torch_fallback":
            return "pytorch_not_available"
        if status == "trained":
            return "trained_policy_inference_not_enabled_yet"
        return "ppo_checkpoint_policy_not_available"
