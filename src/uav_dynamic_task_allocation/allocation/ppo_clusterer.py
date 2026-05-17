"""PPO policy adapter for dynamic target regrouping."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from uav_dynamic_task_allocation.algorithms.rl.ppo.checkpoint import (
    PPOCheckpointError,
    load_ppo_checkpoint,
    load_ppo_metadata,
)
from uav_dynamic_task_allocation.algorithms.rl.ppo.network import (
    PPOActorCritic,
    PPONetworkConfig,
    torch,
)
from uav_dynamic_task_allocation.allocation.pso_clusterer import (
    PSOClusterer,
    PSOClustererConfig,
)
from uav_dynamic_task_allocation.envs.target_grouping_env import (
    TargetGroupingEnv,
    TargetGroupingEnvConfig,
)


class PPOClustererError(Exception):
    """Raised when PPO target regrouping cannot be executed."""


@dataclass(frozen=True)
class PPOClustererConfig:
    """Configuration for PPO-based dynamic target regrouping."""

    checkpoint_dir: str = "checkpoints/ppo_clusterer"
    fallback_method: str = "pso"
    random_seed: int = 42
    max_inference_steps: int = 30
    inference_mode: str = "greedy"
    max_targets_per_cluster: int = 64
    env_config: dict[str, Any] = field(default_factory=dict)
    pso_config: PSOClustererConfig = field(default_factory=PSOClustererConfig)


@dataclass
class PPOClustererResult:
    """Dynamic target regrouping result."""

    labels: np.ndarray
    centers: np.ndarray
    method_used: str
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str | None = None
    num_inference_steps: int = 0


class PPOClusterer:
    """
    Load a trained PPO policy for dynamic regrouping, with PSO fallback.
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
        features = np.asarray(features, dtype=np.float64)
        target_values = (
            np.asarray(target_values, dtype=np.float64)
            if target_values is not None
            else np.ones(len(features), dtype=np.float64)
        )
        target_defenses = (
            np.asarray(target_defenses, dtype=np.float64)
            if target_defenses is not None
            else np.ones(len(features), dtype=np.float64)
        )

        pso_result = self._run_pso(
            features=features,
            num_clusters=num_clusters,
            target_values=target_values,
            target_defenses=target_defenses,
        )

        try:
            metadata = load_ppo_metadata(self.config.checkpoint_dir)
            if metadata.get("status") != "trained":
                raise PPOClustererError(
                    f"PPO checkpoint status is not trained: {metadata.get('status')}"
                )
            result = self._run_policy(
                features=features,
                num_clusters=num_clusters,
                target_values=target_values,
                target_defenses=target_defenses,
                initial_labels=pso_result.labels,
                metadata=metadata,
            )
            return result
        except Exception as exc:
            return self._fallback_result(
                pso_result=pso_result,
                reason=str(exc),
            )

    def _run_policy(
        self,
        features: np.ndarray,
        num_clusters: int,
        target_values: np.ndarray,
        target_defenses: np.ndarray,
        initial_labels: np.ndarray,
        metadata: dict[str, Any],
    ) -> PPOClustererResult:
        if torch is None:
            raise PPOClustererError("PyTorch is not available.")

        checkpoint = load_ppo_checkpoint(self.config.checkpoint_dir, map_location="cpu")
        model_metadata = checkpoint["metadata"]
        env_config = self._env_config_from_metadata(model_metadata, num_clusters)
        env = TargetGroupingEnv(
            features=features,
            labels=initial_labels,
            config=env_config,
            target_values=target_values,
            target_defenses=target_defenses,
        )
        observation = env.reset()

        network_config = PPONetworkConfig(
            observation_dim=int(model_metadata.get("observation_dim", env.observation_dim)),
            action_dim=int(model_metadata.get("action_dim", env.action_dim)),
            hidden_dims=tuple(int(value) for value in model_metadata.get("hidden_dims", [128, 128])),
        )
        if network_config.observation_dim != env.observation_dim:
            raise PPOClustererError(
                "PPO observation_dim mismatch: "
                f"checkpoint={network_config.observation_dim}, env={env.observation_dim}"
            )
        if network_config.action_dim != env.action_dim:
            raise PPOClustererError(
                "PPO action_dim mismatch: "
                f"checkpoint={network_config.action_dim}, env={env.action_dim}"
            )

        model = PPOActorCritic(network_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        inference_steps = 0
        rng = np.random.default_rng(self.config.random_seed)

        while inference_steps < self.config.max_inference_steps:
            action_mask = env.action_mask()
            if not action_mask.any():
                break
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                logits = logits.masked_fill(~mask_tensor, -1e9)
                if self.config.inference_mode == "sample":
                    dist = torch.distributions.Categorical(logits=logits)
                    action = int(dist.sample().item())
                else:
                    action = int(torch.argmax(logits, dim=-1).item())
            if action_mask[action] <= 0:
                legal_actions = np.where(action_mask > 0)[0]
                action = int(rng.choice(legal_actions))
            observation, _, done, _ = env.step_index(action)
            inference_steps += 1
            if done:
                break

        labels = env.get_current_labels()
        centers = self._recompute_centers(features, labels, num_clusters)
        metrics = env.get_metrics()
        metrics["mean_distance_to_center"] = metrics["mean_intra_cluster_distance"]

        return PPOClustererResult(
            labels=labels,
            centers=centers,
            method_used="ppo_policy",
            metrics=metrics,
            metadata={
                **metadata,
                "checkpoint_metadata": model_metadata,
                "num_inference_steps": inference_steps,
            },
            fallback_reason=None,
            num_inference_steps=inference_steps,
        )

    def _run_pso(
        self,
        features: np.ndarray,
        num_clusters: int,
        target_values: np.ndarray,
        target_defenses: np.ndarray,
    ):
        if self.config.fallback_method != "pso":
            raise PPOClustererError(
                f"Unsupported PPO fallback_method={self.config.fallback_method}"
            )
        return PSOClusterer(self.config.pso_config).cluster(
            features=features,
            num_clusters=num_clusters,
            target_values=target_values,
            target_defenses=target_defenses,
        )

    def _fallback_result(self, pso_result, reason: str) -> PPOClustererResult:
        metrics = dict(pso_result.metrics)
        metrics.setdefault(
            "mean_intra_cluster_distance",
            metrics.get("mean_distance_to_center", 0.0),
        )
        return PPOClustererResult(
            labels=pso_result.labels,
            centers=pso_result.centers,
            method_used="ppo_fallback_pso",
            metrics=metrics,
            metadata={
                "fallback_method": "pso",
                "fallback_reason": reason,
                "reason": reason,
                "pso_metadata": pso_result.metadata,
            },
            fallback_reason=reason,
            num_inference_steps=0,
        )

    def _env_config_from_metadata(
        self,
        metadata: dict[str, Any],
        num_clusters: int,
    ) -> TargetGroupingEnvConfig:
        saved_env = dict(metadata.get("env_config", {}))
        saved_env.update(self.config.env_config)
        return TargetGroupingEnvConfig(
            num_clusters=num_clusters,
            max_steps=int(saved_env.get("max_steps", self.config.max_inference_steps)),
            max_targets_per_cluster=int(
                saved_env.get("max_targets_per_cluster", self.config.max_targets_per_cluster)
            ),
            illegal_action_penalty=float(saved_env.get("illegal_action_penalty", -1.0)),
            count_balance_weight=float(saved_env.get("count_balance_weight", 1.0)),
            compactness_weight=float(saved_env.get("compactness_weight", 1.0)),
            workload_balance_weight=float(saved_env.get("workload_balance_weight", 1.0)),
            empty_cluster_penalty=float(saved_env.get("empty_cluster_penalty", -1.0)),
            move_penalty=float(saved_env.get("move_penalty", -0.01)),
            improvement_bonus=float(saved_env.get("improvement_bonus", 0.05)),
            patience=int(saved_env.get("patience", 8)),
        )

    @staticmethod
    def _recompute_centers(
        features: np.ndarray,
        labels: np.ndarray,
        num_clusters: int,
    ) -> np.ndarray:
        centers = np.zeros((num_clusters, features.shape[1]), dtype=np.float64)
        for cluster_id in range(num_clusters):
            members = features[labels == cluster_id]
            if len(members) > 0:
                centers[cluster_id] = members.mean(axis=0)
        return centers
