"""训练PPO 算法clusterer脚本，封装可直接运行的实验、检查或可视化流程。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_dynamic_task_allocation.algorithms.rl.ppo.agent import PPOAgentConfig
from uav_dynamic_task_allocation.algorithms.rl.ppo.checkpoint import save_ppo_checkpoint
from uav_dynamic_task_allocation.algorithms.rl.ppo.metrics import write_ppo_metrics
from uav_dynamic_task_allocation.algorithms.rl.ppo.network import torch
from uav_dynamic_task_allocation.algorithms.rl.ppo.trainer import (
    PPOClustererTrainer,
    PPOTrainerConfig,
)
from uav_dynamic_task_allocation.allocation.ppo_clusterer import (
    PPOClusterer,
    PPOClustererConfig,
)
from uav_dynamic_task_allocation.envs.target_grouping_env import (
    TargetGroupingEnv,
    TargetGroupingEnvConfig,
)
from uav_dynamic_task_allocation.utils.config import get_project_root, load_and_validate_config


def _build_training_env(config: dict) -> TargetGroupingEnv:
    """构建后续流程需要的领域对象或配置对象，处理训练环境相关数据。

    参数：
        config: 配置，类型为 dict。

    返回：
        TargetGroupingEnv，表示该函数计算或构建得到的结果。
    """
    ppo_config = config.get("ppo", {})
    features = np.asarray(
        [
            [0.05, 0.05],
            [0.12, 0.08],
            [0.18, 0.14],
            [0.82, 0.78],
            [0.90, 0.85],
            [0.75, 0.93],
            [0.48, 0.42],
            [0.52, 0.58],
            [0.44, 0.53],
        ],
        dtype=float,
    )
    labels = np.asarray([0, 0, 2, 1, 1, 1, 2, 2, 0], dtype=np.int64)
    values = np.asarray([9, 7, 8, 6, 5, 7, 10, 8, 6], dtype=float)
    defenses = np.asarray([3, 2, 4, 2, 3, 3, 5, 4, 2], dtype=float)
    env_config = TargetGroupingEnvConfig(
        num_clusters=3,
        max_steps=int(ppo_config.get("max_steps", 30)),
        max_targets_per_cluster=int(ppo_config.get("max_targets_per_cluster", 16)),
        illegal_action_penalty=float(ppo_config.get("illegal_action_penalty", -1.0)),
        count_balance_weight=float(ppo_config.get("count_balance_weight", 1.0)),
        compactness_weight=float(ppo_config.get("compactness_weight", 1.0)),
        workload_balance_weight=float(ppo_config.get("workload_balance_weight", 1.0)),
        dynamic_event_bonus=float(ppo_config.get("dynamic_event_bonus", 0.05)),
    )
    return TargetGroupingEnv(
        features=features,
        labels=labels,
        config=env_config,
        target_values=values,
        target_defenses=defenses,
    )


def _checkpoint_dir(project_root: Path, config: dict) -> Path:
    """处理检查点dir相关业务逻辑。

    参数：
        project_root: projectroot，类型为 Path。
        config: 配置，类型为 dict。

    返回：
        Path，表示该函数计算或构建得到的结果。
    """
    raw_path = Path(config.get("ppo", {}).get("checkpoint_dir", "checkpoints/ppo_clusterer"))
    return raw_path if raw_path.is_absolute() else project_root / raw_path


def _run_no_torch_fallback(project_root: Path, config: dict, env: TargetGroupingEnv) -> Path:
    """执行对应的流程步骤并返回运行结果，处理notorchfallback相关数据。

    参数：
        project_root: projectroot，类型为 Path。
        config: 配置，类型为 dict。
        env: 环境，类型为 TargetGroupingEnv。

    返回：
        Path，表示该函数计算或构建得到的结果。
    """
    result = PPOClusterer(
        PPOClustererConfig(
            checkpoint_dir=str(_checkpoint_dir(project_root, config)),
            random_seed=int(config.get("project", {}).get("seed", 42)),
        )
    ).cluster(
        features=env.features[:, :2],
        num_clusters=env.config.num_clusters,
        target_values=env.values,
        target_defenses=env.defenses,
    )
    metrics_path = project_root / "outputs" / "metrics" / "ppo_clusterer_training_metrics.csv"
    write_ppo_metrics(
        [
            {
                "episode": 0,
                "total_reward": env.get_metrics()["score"],
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "method_used": result.method_used,
            }
        ],
        metrics_path,
    )
    checkpoint_path = save_ppo_checkpoint(
        _checkpoint_dir(project_root, config),
        metadata={
            "algorithm": "ppo_clusterer",
            "status": "no_torch_fallback",
            "method_used": result.method_used,
            "metrics_csv": str(metrics_path),
            "labels": result.labels.tolist(),
            "metrics": result.metrics,
            "note": "PyTorch is not installed; dynamic regrouping falls back to PSO.",
        },
    )
    return checkpoint_path


def main() -> None:
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        无返回值；通过状态变更、文件输出或日志记录体现执行结果。
    """
    project_root = get_project_root()
    config = load_and_validate_config(project_root / "configs" / "default.yaml")
    env = _build_training_env(config)
    ppo_config = config.get("ppo", {})

    if torch is None:
        checkpoint_path = _run_no_torch_fallback(project_root, config, env)
        print(f"PPO clusterer fallback checkpoint saved to: {checkpoint_path}")
        return

    agent_config = PPOAgentConfig(
        learning_rate=float(ppo_config.get("learning_rate", 3e-4)),
        gamma=float(ppo_config.get("gamma", 0.99)),
        gae_lambda=float(ppo_config.get("gae_lambda", 0.95)),
        clip_epsilon=float(ppo_config.get("clip_epsilon", 0.2)),
        entropy_coef=float(ppo_config.get("entropy_coef", 0.01)),
        value_loss_coef=float(ppo_config.get("value_loss_coef", 0.5)),
        batch_size=int(ppo_config.get("batch_size", 64)),
        update_epochs=int(ppo_config.get("update_epochs", 4)),
    )
    trainer = PPOClustererTrainer(
        env=env,
        agent_config=agent_config,
        trainer_config=PPOTrainerConfig(
            max_episodes=int(ppo_config.get("max_episodes", 20)),
            seed=int(config.get("project", {}).get("seed", 42)),
            device=str(config.get("project", {}).get("device", "cpu")),
        ),
    )
    rows, agent = trainer.train()
    metrics_path = project_root / "outputs" / "metrics" / "ppo_clusterer_training_metrics.csv"
    write_ppo_metrics(rows, metrics_path)
    checkpoint_path = save_ppo_checkpoint(
        _checkpoint_dir(project_root, config),
        metadata={
            "algorithm": "ppo_clusterer",
            "status": "trained",
            "metrics_csv": str(metrics_path),
            "episodes": len(rows),
            "env_metrics": env.get_metrics(),
        },
        model_state=agent.network.state_dict(),
    )
    print(f"PPO clusterer checkpoint saved to: {checkpoint_path}")
    print(f"PPO clusterer training metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
