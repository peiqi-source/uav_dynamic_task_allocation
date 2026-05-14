"""资源分配模块中的PPO 算法clusterer实现。"""
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

    # checkpoint_dir: 检查点dir。
    checkpoint_dir: str = "checkpoints/ppo_clusterer"
    # fallback_method: fallbackmethod。
    fallback_method: str = "pso"
    # random_seed: 随机随机种子。
    random_seed: int = 42
    # pso_config: PSO 算法配置。
    pso_config: PSOClustererConfig = field(default_factory=PSOClustererConfig)


@dataclass
class PPOClustererResult:
    """Dynamic target regrouping result."""

    # labels: labels 数据。
    labels: np.ndarray
    # centers: centers 数据。
    centers: np.ndarray
    # method_used: methodused。
    method_used: str
    # metrics: 指标集合。
    metrics: dict[str, Any] = field(default_factory=dict)
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)


class PPOClusterer:
    """
    PPO dynamic regrouping adapter.

    This class provides the formal engineering entry for dynamic regrouping.
    If a trained PPO checkpoint is not present, it falls back to PSO so the
    event-driven mission pipeline remains runnable and reproducible.
    """

    def __init__(self, config: PPOClustererConfig) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            config: 配置对象，类型为 PPOClustererConfig。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        # config: 配置。
        self.config = config

    def cluster(
        self,
        features: np.ndarray,
        num_clusters: int,
        target_values: np.ndarray | None = None,
        target_defenses: np.ndarray | None = None,
    ) -> PPOClustererResult:
        """处理目标簇相关业务逻辑。

        参数：
            features: features 数据，类型为 np.ndarray。
            num_clusters: num目标簇集合，类型为 int。
            target_values: 目标values，类型为 np.ndarray | None。
            target_defenses: 目标defenses，类型为 np.ndarray | None。

        返回：
            PPOClustererResult，表示该函数计算或构建得到的结果。
        """
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
        """处理检查点路径相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            Path，表示该函数计算或构建得到的结果。
        """
        return Path(self.config.checkpoint_dir) / "ppo_clusterer_checkpoint.json"

    @staticmethod
    def _fallback_reason(metadata: dict[str, Any]) -> str:
        """处理fallbackreason相关业务逻辑。

        参数：
            metadata: 扩展元数据，类型为 dict[str, Any]。

        返回：
            str，表示该函数计算或构建得到的结果。
        """
        status = metadata.get("status")
        if status == "no_torch_fallback":
            return "pytorch_not_available"
        if status == "trained":
            return "trained_policy_inference_not_enabled_yet"
        return "ppo_checkpoint_policy_not_available"
