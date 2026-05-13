"""训练打击顺序DQN 算法脚本，封装可直接运行的实验、检查或可视化流程。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import torch

    from uav_dynamic_task_allocation.algorithms.rl.dqn.agent import (
        DQNAgent,
        DQNAgentConfig,
    )
    from uav_dynamic_task_allocation.algorithms.rl.dqn.checkpoint import (
        DQNCheckpointConfig,
        DQNCheckpointManager,
    )
    from uav_dynamic_task_allocation.algorithms.rl.dqn.metrics import (
        DQNMetricsConfig,
        DQNMetricsWriter,
    )
    from uav_dynamic_task_allocation.algorithms.rl.dqn.network import (
        DQNNetworkConfig,
    )
    from uav_dynamic_task_allocation.algorithms.rl.dqn.replay_buffer import ReplayBuffer
    from uav_dynamic_task_allocation.algorithms.rl.dqn.strike_order_trainer import (
        StrikeOrderDQNTrainer,
        load_strike_order_dqn_trainer_config,
    )
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False
from uav_dynamic_task_allocation.core.contracts import TargetCluster
from uav_dynamic_task_allocation.core.entities import Position, build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.envs.strike_order_env import (
    StrikeOrderEnv,
    load_strike_order_env_config,
)
from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.device import get_device
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config
from uav_dynamic_task_allocation.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train StrikeOrder DQN for target-cluster strike ordering."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override strike_order_dqn.trainer.num_episodes.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override experiment.device.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override experiment.name.",
    )

    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint saving.",
    )

    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable metrics saving.",
    )

    return parser.parse_args()


def set_nested_config_value(
    config: dict[str, Any],
    key_path: str,
    value: Any,
) -> None:
    """修改嵌套配置字典中的某个值。"""
    keys = key_path.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def resolve_config_path(config_path: str) -> Path:
    """解析配置路径。"""
    path = Path(config_path)

    if path.is_absolute():
        return path

    return get_project_root() / path


def apply_command_line_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """用命令行参数临时覆盖 YAML 配置。"""
    if args.episodes is not None:
        set_nested_config_value(
            config,
            "strike_order_dqn.trainer.num_episodes",
            args.episodes,
        )

    if args.device is not None:
        set_nested_config_value(
            config,
            "experiment.device",
            args.device,
        )

    if args.experiment_name is not None:
        set_nested_config_value(
            config,
            "experiment.name",
            args.experiment_name,
        )

    if args.no_checkpoint:
        set_nested_config_value(
            config,
            "strike_order_dqn.trainer.save_checkpoint",
            False,
        )

    if args.no_metrics:
        set_nested_config_value(
            config,
            "strike_order_dqn.metrics.enabled",
            False,
        )


def build_debug_clusters(
    targets,
    num_clusters: int,
    targets_per_cluster: int,
) -> list[TargetCluster]:
    """
    用真实 Target 数据临时构造 debug clusters。

    这是调试阶段使用的临时分群方法。
    后面 target_clustering.py 完成后，会替换为 PSO / PPO / KMeans 输出的 TargetClusterSet。

    当前版本会根据实际目标数量自动调整，避免因为配置的
    num_clusters * targets_per_cluster 大于真实目标数量而直接报错。
    """
    if num_clusters <= 0:
        raise RuntimeError("num_clusters must be positive.")

    if targets_per_cluster <= 0:
        raise RuntimeError("targets_per_cluster must be positive.")

    if not targets:
        raise RuntimeError("No targets available to build debug clusters.")

    total_targets = len(targets)

    # 调试阶段最多只能使用已有目标数量。
    requested_targets = num_clusters * targets_per_cluster
    used_targets = min(requested_targets, total_targets)

    # 根据实际可用目标数重新计算 cluster 数量。
    # 例如 20 个目标，原本要求 5*6=30，则自动使用 4 个 cluster，每个约 5 个目标。
    actual_num_clusters = min(num_clusters, used_targets)

    if actual_num_clusters <= 0:
        raise RuntimeError("actual_num_clusters must be positive.")

    selected_targets = targets[:used_targets]

    clusters: list[TargetCluster] = []

    # np.array_split 可以把目标尽量均匀地切成多个 cluster。
    import numpy as np

    target_indices = np.array_split(
        np.arange(len(selected_targets)),
        actual_num_clusters,
    )

    for cluster_id, indices in enumerate(target_indices):
        cluster_targets = [selected_targets[int(index)] for index in indices]

        if not cluster_targets:
            continue

        center = Position(
            x=sum(target.position.x for target in cluster_targets)
            / len(cluster_targets),
            y=sum(target.position.y for target in cluster_targets)
            / len(cluster_targets),
        )

        cluster = TargetCluster(
            cluster_id=cluster_id,
            targets=cluster_targets,
            center=center,
            defense_sum=sum(target.defense for target in cluster_targets),
            significance_sum=sum(target.significance for target in cluster_targets),
            compactness=None,
            metadata={
                "source": "debug_cluster_builder",
                "requested_num_clusters": num_clusters,
                "requested_targets_per_cluster": targets_per_cluster,
                "actual_num_clusters": actual_num_clusters,
                "actual_num_targets": len(cluster_targets),
                "total_available_targets": total_targets,
            },
        )
        cluster.validate()
        clusters.append(cluster)

    if not clusters:
        raise RuntimeError("No debug clusters were created.")

    return clusters


def load_strike_order_network_config(
    config: dict[str, Any],
    input_dim: int,
    action_dim: int,
) -> DQNNetworkConfig:
    """
    读取 StrikeOrder DQN 网络配置。

    这里单独读取 strike_order_dqn.network，
    避免和之前的全局 dqn.network 混在一起。
    """
    prefix = "strike_order_dqn.network"

    hidden_dims_raw = get_config_value(
        config,
        f"{prefix}.hidden_dims",
        default=[256, 256],
    )

    network_config = DQNNetworkConfig(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dims=tuple(int(value) for value in hidden_dims_raw),
        activation=str(
            get_config_value(config, f"{prefix}.activation", default="relu")
        ),
        dropout=float(
            get_config_value(config, f"{prefix}.dropout", default=0.0)
        ),
        use_layer_norm=bool(
            get_config_value(config, f"{prefix}.use_layer_norm", default=False)
        ),
        network_type=str(
            get_config_value(config, f"{prefix}.network_type", default="dqn")
        ),
    )

    network_config.validate()
    return network_config


def load_strike_order_agent_config(
    config: dict[str, Any],
) -> DQNAgentConfig:
    """读取 StrikeOrder DQN agent 配置。"""
    prefix = "strike_order_dqn.agent"

    agent_config = DQNAgentConfig(
        gamma=float(get_config_value(config, f"{prefix}.gamma", default=0.99)),
        learning_rate=float(
            get_config_value(config, f"{prefix}.learning_rate", default=5e-4)
        ),
        batch_size=int(
            get_config_value(config, f"{prefix}.batch_size", default=32)
        ),
        target_update_interval=int(
            get_config_value(
                config,
                f"{prefix}.target_update_interval",
                default=100,
            )
        ),
        epsilon_start=float(
            get_config_value(config, f"{prefix}.epsilon_start", default=1.0)
        ),
        epsilon_end=float(
            get_config_value(config, f"{prefix}.epsilon_end", default=0.05)
        ),
        epsilon_decay_steps=int(
            get_config_value(
                config,
                f"{prefix}.epsilon_decay_steps",
                default=3000,
            )
        ),
        gradient_clip_norm=float(
            get_config_value(
                config,
                f"{prefix}.gradient_clip_norm",
                default=10.0,
            )
        ),
        optimizer=str(
            get_config_value(config, f"{prefix}.optimizer", default="adam")
        ),
    )

    agent_config.validate()
    return agent_config


def load_strike_order_checkpoint_config(
    config: dict[str, Any],
) -> DQNCheckpointConfig:
    """读取 StrikeOrder DQN checkpoint 配置。"""
    prefix = "strike_order_dqn.checkpoint"

    checkpoint_config = DQNCheckpointConfig(
        enabled=bool(
            get_config_value(config, f"{prefix}.enabled", default=True)
        ),
        checkpoint_dir=str(
            get_config_value(
                config,
                f"{prefix}.checkpoint_dir",
                default="checkpoints/strike_order_dqn",
            )
        ),
        latest_filename=str(
            get_config_value(
                config,
                f"{prefix}.latest_filename",
                default="latest.pt",
            )
        ),
        best_filename=str(
            get_config_value(
                config,
                f"{prefix}.best_filename",
                default="best.pt",
            )
        ),
        save_optimizer=bool(
            get_config_value(
                config,
                f"{prefix}.save_optimizer",
                default=True,
            )
        ),
        save_replay_buffer=bool(
            get_config_value(
                config,
                f"{prefix}.save_replay_buffer",
                default=False,
            )
        ),
    )

    checkpoint_config.validate()
    return checkpoint_config


def load_strike_order_metrics_config(
    config: dict[str, Any],
) -> DQNMetricsConfig:
    """读取 StrikeOrder DQN metrics 配置。"""
    prefix = "strike_order_dqn.metrics"

    metrics_config = DQNMetricsConfig(
        enabled=bool(
            get_config_value(config, f"{prefix}.enabled", default=True)
        ),
        metrics_dir=str(
            get_config_value(
                config,
                f"{prefix}.metrics_dir",
                default="outputs/metrics",
            )
        ),
        metrics_filename=str(
            get_config_value(
                config,
                f"{prefix}.metrics_filename",
                default="strike_order_dqn_metrics.csv",
            )
        ),
        overwrite=bool(
            get_config_value(config, f"{prefix}.overwrite", default=True)
        ),
        save_jsonl=bool(
            get_config_value(config, f"{prefix}.save_jsonl", default=True)
        ),
        jsonl_filename=str(
            get_config_value(
                config,
                f"{prefix}.jsonl_filename",
                default="strike_order_dqn_metrics.jsonl",
            )
        ),
    )

    metrics_config.validate()
    return metrics_config


def run_no_torch_baseline(
    config: dict[str, Any],
    project_root: Path,
    logger,
) -> None:
    """Run a deterministic nearest-neighbor baseline when PyTorch is unavailable."""
    logger.warning(
        "PyTorch is not installed. Skipping neural DQN optimization and running "
        "a deterministic nearest-neighbor strike-order baseline instead."
    )

    data = load_all_data(config)
    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )
    env_config = load_strike_order_env_config(config)
    clusters = build_debug_clusters(
        targets=state.targets,
        num_clusters=int(
            get_config_value(
                config,
                "strike_order_dqn.cluster_dataset.num_debug_clusters",
                default=5,
            )
        ),
        targets_per_cluster=int(
            get_config_value(
                config,
                "strike_order_dqn.cluster_dataset.targets_per_cluster",
                default=6,
            )
        ),
    )

    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        env = StrikeOrderEnv(
            target_cluster=cluster,
            start_position=cluster.center,
            config=env_config,
        )
        observation = env.reset()
        total_reward = 0.0
        while int(observation.action_mask.sum()) > 0:
            action_id = _select_nearest_action(env, observation.action_mask)
            step_result = env.step(action_id)
            total_reward += float(step_result.reward)
            observation = step_result.observation
            if step_result.done:
                break
        plan_dict = env.get_strike_order_plan_dict()
        rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "method": "nearest_neighbor_no_torch_baseline",
                "total_reward": total_reward,
                "total_path_distance": plan_dict["total_path_distance"],
                "ordered_target_ids": plan_dict["ordered_target_ids"],
            }
        )

    metrics_path = project_root / "outputs" / "metrics" / "strike_order_dqn_no_torch_baseline.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_path = project_root / "checkpoints" / "strike_order_dqn" / "no_torch_baseline.json"
    network_type = str(
        get_config_value(config, "strike_order_dqn.network.network_type", default="dqn")
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint_type": "no_torch_baseline",
                "method": "nearest_neighbor",
                "requested_network_type": network_type,
                "num_clusters": len(rows),
                "metrics_csv": str(metrics_path),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"No-torch baseline metrics CSV: {metrics_path}")
    logger.info(f"No-torch baseline checkpoint metadata: {checkpoint_path}")


def _select_nearest_action(env: StrikeOrderEnv, action_mask) -> int:
    """按照策略从候选集合中选择目标对象，处理nearest动作相关数据。

    参数：
        env: 环境，类型为 StrikeOrderEnv。
        action_mask: 动作掩码。

    返回：
        int，表示该函数计算或构建得到的结果。
    """
    valid_action_ids = [
        action_id for action_id, is_valid in enumerate(action_mask)
        if bool(is_valid)
    ]
    if not valid_action_ids:
        raise RuntimeError("No valid action available.")
    return min(
        valid_action_ids,
        key=lambda action_id: env.current_position.distance_to(
            env.target_cluster.targets[action_id].position
        ),
    )


def main() -> None:
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        无返回值；通过状态变更、文件输出或日志记录体现执行结果。
    """
    args = parse_args()

    project_root = get_project_root()
    config_path = resolve_config_path(args.config)

    config = load_and_validate_config(config_path)
    apply_command_line_overrides(config, args)

    logger = setup_logger_from_config(config)

    logger.info("=" * 80)
    logger.info("StrikeOrder DQN training entry started.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config path: {config_path}")
    logger.info("=" * 80)

    if not TORCH_AVAILABLE:
        run_no_torch_baseline(
            config=config,
            project_root=project_root,
            logger=logger,
        )
        logger.info("StrikeOrder DQN training entry finished with no-torch baseline.")
        return

    seed = int(get_config_value(config, "experiment.seed", default=42))
    preferred_device = str(
        get_config_value(config, "experiment.device", default="auto")
    )

    set_seed(seed)

    device = torch.device(get_device(preferred_device))

    logger.info(f"Seed: {seed}")
    logger.info(f"Selected device: {device}")

    data = load_all_data(config)

    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    strike_order_env_config = load_strike_order_env_config(config)
    trainer_config = load_strike_order_dqn_trainer_config(config)
    agent_config = load_strike_order_agent_config(config)
    checkpoint_config = load_strike_order_checkpoint_config(config)
    metrics_config = load_strike_order_metrics_config(config)

    num_debug_clusters = int(
        get_config_value(
            config,
            "strike_order_dqn.cluster_dataset.num_debug_clusters",
            default=5,
        )
    )
    targets_per_cluster = int(
        get_config_value(
            config,
            "strike_order_dqn.cluster_dataset.targets_per_cluster",
            default=6,
        )
    )

    clusters = build_debug_clusters(
        targets=state.targets,
        num_clusters=num_debug_clusters,
        targets_per_cluster=targets_per_cluster,
    )

    logger.info(
        "Debug clusters built: "
        f"num_clusters={len(clusters)}, "
        f"targets_per_cluster={targets_per_cluster}"
    )

    # 用第一个 cluster 构造一个 sample env，计算 DQN 输入输出维度。
    sample_env = StrikeOrderEnv(
        target_cluster=clusters[0],
        start_position=clusters[0].center,
        config=strike_order_env_config,
    )
    sample_observation = sample_env.reset()

    input_dim = int(sample_observation.vector.shape[0])
    action_dim = sample_env.action_dim

    network_config = load_strike_order_network_config(
        config=config,
        input_dim=input_dim,
        action_dim=action_dim,
    )

    logger.info(f"StrikeOrder input dim: {input_dim}")
    logger.info(f"StrikeOrder action dim: {action_dim}")
    logger.info(f"Network config: {network_config}")
    logger.info(f"Agent config: {agent_config}")
    logger.info(f"Trainer config: {trainer_config}")

    agent = DQNAgent(
        network_config=network_config,
        agent_config=agent_config,
        device=device,
        seed=seed,
    )

    replay_buffer = ReplayBuffer(
        capacity=trainer_config.replay_buffer_capacity,
        seed=seed,
    )

    checkpoint_manager = None
    if trainer_config.save_checkpoint:
        checkpoint_manager = DQNCheckpointManager(
            config=checkpoint_config,
            project_root=project_root,
        )
        logger.info(
            "StrikeOrder checkpoint manager initialized: "
            f"dir={checkpoint_manager.checkpoint_dir}"
        )

    metrics_writer = None
    if metrics_config.enabled:
        metrics_writer = DQNMetricsWriter(
            config=metrics_config,
            project_root=project_root,
        )
        logger.info(
            "StrikeOrder metrics writer initialized: "
            f"csv={metrics_writer.csv_path}, "
            f"jsonl={metrics_writer.jsonl_path if metrics_config.save_jsonl else None}"
        )

    trainer = StrikeOrderDQNTrainer(
        clusters=clusters,
        strike_order_env_config=strike_order_env_config,
        agent=agent,
        replay_buffer=replay_buffer,
        trainer_config=trainer_config,
        checkpoint_manager=checkpoint_manager,
        metrics_writer=metrics_writer,
        logger=logger,
        seed=seed,
    )

    result = trainer.train()

    logger.info("=" * 80)
    logger.info("StrikeOrder DQN training entry finished.")
    logger.info(f"Best episode: {result.best_episode}")
    logger.info(f"Best reward: {result.best_reward}")
    logger.info(f"Last reward: {result.last_reward}")
    logger.info(f"Total environment steps: {trainer.global_env_steps}")
    logger.info(f"Replay buffer size: {len(replay_buffer)}")
    logger.info(f"Final epsilon: {agent.get_epsilon():.6f}")

    if checkpoint_manager is not None:
        logger.info(f"Latest checkpoint: {checkpoint_manager.latest_path}")
        logger.info(f"Best checkpoint: {checkpoint_manager.best_path}")

    if metrics_writer is not None:
        logger.info(f"Metrics CSV: {metrics_writer.csv_path}")
        if metrics_config.save_jsonl:
            logger.info(f"Metrics JSONL: {metrics_writer.jsonl_path}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
