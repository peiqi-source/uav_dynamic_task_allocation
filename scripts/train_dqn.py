from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from uav_dynamic_task_allocation.algorithms.rl.dqn.agent import (
    DQNAgent,
    load_dqn_agent_config,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.checkpoint import (
    DQNCheckpointManager,
    load_dqn_checkpoint_config,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.metrics import (
    DQNMetricsWriter,
    load_dqn_metrics_config,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.network import (
    load_dqn_network_config,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.replay_buffer import ReplayBuffer
from uav_dynamic_task_allocation.algorithms.rl.dqn.trainer import (
    DQNTrainer,
    load_dqn_trainer_config,
)
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.envs.action_space import (
    FixedUAVTargetActionSpace,
    load_action_space_config,
)
from uav_dynamic_task_allocation.envs.drone_battle_env import DroneBattleEnv
from uav_dynamic_task_allocation.envs.env_config import load_env_config
from uav_dynamic_task_allocation.envs.observation import (
    ObservationBuilder,
    load_observation_config,
)
from uav_dynamic_task_allocation.envs.reward import load_reward_config
from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.device import get_device
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config
from uav_dynamic_task_allocation.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    这里允许从命令行临时覆盖少量常用训练参数，
    这样你不用每次都手动修改 default.yaml。

    例如：
        python scripts/train_dqn.py --episodes 100
        python scripts/train_dqn.py --device cpu
        python scripts/train_dqn.py --no-checkpoint
    """
    parser = argparse.ArgumentParser(
        description="Train DQN for UAV dynamic task allocation."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override dqn.trainer.num_episodes.",
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
        help="Disable checkpoint saving for this run.",
    )

    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable metrics saving for this run.",
    )

    return parser.parse_args()


def set_nested_config_value(
    config: dict[str, Any],
    key_path: str,
    value: Any,
) -> None:
    """
    修改嵌套配置字典中的某个值。

    例如：
        set_nested_config_value(config, "dqn.trainer.num_episodes", 100)

    会修改：
        config["dqn"]["trainer"]["num_episodes"] = 100

    这个函数只用于命令行临时覆盖配置。
    正式实验的主配置仍然建议写在 YAML 文件里。
    """
    keys = key_path.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def resolve_config_path(config_path: str) -> Path:
    """
    解析配置文件路径。

    如果传入的是相对路径，则默认相对于项目根目录。
    如果传入的是绝对路径，则直接使用。
    """
    path = Path(config_path)

    if path.is_absolute():
        return path

    return get_project_root() / path


def apply_command_line_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """
    根据命令行参数临时覆盖配置。

    注意：
    这里不会写回 default.yaml，只影响当前这一次运行。
    """
    if args.episodes is not None:
        set_nested_config_value(
            config,
            "dqn.trainer.num_episodes",
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
            "dqn.trainer.save_checkpoint",
            False,
        )

    if args.no_metrics:
        set_nested_config_value(
            config,
            "dqn.metrics.enabled",
            False,
        )


def main() -> None:
    args = parse_args()

    project_root = get_project_root()
    config_path = resolve_config_path(args.config)

    config = load_and_validate_config(config_path)
    apply_command_line_overrides(config, args)

    logger = setup_logger_from_config(config)

    logger.info("=" * 80)
    logger.info("DQN training entry started.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config path: {config_path}")
    logger.info("=" * 80)

    seed = int(get_config_value(config, "experiment.seed", default=42))
    preferred_device = str(
        get_config_value(config, "experiment.device", default="auto")
    )

    set_seed(seed)

    device = torch.device(get_device(preferred_device))

    logger.info(f"Seed: {seed}")
    logger.info(f"Selected device: {device}")

    # ------------------------------------------------------------------
    # 1. 加载数据和配置
    # ------------------------------------------------------------------
    logger.info("Loading data and configuration.")

    data = load_all_data(config)

    env_config = load_env_config(config)
    reward_config = load_reward_config(config)
    obs_config = load_observation_config(config)
    action_space_config = load_action_space_config(config)

    agent_config = load_dqn_agent_config(config)
    trainer_config = load_dqn_trainer_config(config)
    checkpoint_config = load_dqn_checkpoint_config(config)
    metrics_config = load_dqn_metrics_config(config)

    logger.info(f"Trainer config: {trainer_config}")
    logger.info(f"Agent config: {agent_config}")
    logger.info(f"Checkpoint enabled: {checkpoint_config.enabled}")
    logger.info(f"Metrics enabled: {metrics_config.enabled}")

    # ------------------------------------------------------------------
    # 2. 构造环境
    # ------------------------------------------------------------------
    logger.info("Building environment.")

    env = DroneBattleEnv(
        uav_df=data["uav"],
        target_df=data["target"],
        env_config=env_config,
        reward_config=reward_config,
    )

    observation_builder = ObservationBuilder(
        env_config=env.env_config,
        obs_config=obs_config,
    )

    action_space = FixedUAVTargetActionSpace(
        env_config=env.env_config,
        action_space_config=action_space_config,
    )

    input_dim = observation_builder.get_input_dim()
    action_dim = action_space.size

    logger.info(f"Observation input dim: {input_dim}")
    logger.info(f"Fixed action dim: {action_dim}")

    # ------------------------------------------------------------------
    # 3. 训练前做一次动作空间快速检查
    # ------------------------------------------------------------------
    env.reset()
    initial_state = env.get_state_copy()
    initial_action_mask = action_space.build_action_mask(initial_state)
    initial_valid_action_count = int(initial_action_mask.sum())

    logger.info(
        "Initial action mask summary: "
        f"{action_space.summarize_mask(initial_state)}"
    )

    if initial_valid_action_count == 0:
        logger.warning(
            "Initial state has no valid action. "
            "Training may stop immediately. "
            "For current pipeline testing, consider setting "
            "action_space.require_reachable=false."
        )

    # ------------------------------------------------------------------
    # 4. 构造 DQN 网络和 Agent
    # ------------------------------------------------------------------
    logger.info("Building DQN agent.")

    network_config = load_dqn_network_config(
        config=config,
        input_dim=input_dim,
        action_dim=action_dim,
    )

    agent = DQNAgent(
        network_config=network_config,
        agent_config=agent_config,
        device=device,
        seed=seed,
    )

    logger.info(f"Network config: {network_config}")
    logger.info(f"Initial epsilon: {agent.get_epsilon():.6f}")

    # ------------------------------------------------------------------
    # 5. 构造 ReplayBuffer
    # ------------------------------------------------------------------
    replay_buffer = ReplayBuffer(
        capacity=trainer_config.replay_buffer_capacity,
        seed=seed,
    )

    logger.info(
        "Replay buffer initialized: "
        f"capacity={replay_buffer.capacity}"
    )

    # ------------------------------------------------------------------
    # 6. 构造 CheckpointManager 和 MetricsWriter
    # ------------------------------------------------------------------
    checkpoint_manager = None
    if trainer_config.save_checkpoint:
        checkpoint_manager = DQNCheckpointManager(
            config=checkpoint_config,
            project_root=project_root,
        )
        logger.info(
            "Checkpoint manager initialized: "
            f"dir={checkpoint_manager.checkpoint_dir}"
        )

    metrics_writer = None
    if metrics_config.enabled:
        metrics_writer = DQNMetricsWriter(
            config=metrics_config,
            project_root=project_root,
        )
        logger.info(
            "Metrics writer initialized: "
            f"csv={metrics_writer.csv_path}, "
            f"jsonl={metrics_writer.jsonl_path if metrics_config.save_jsonl else None}"
        )

    # ------------------------------------------------------------------
    # 7. 构造 Trainer 并开始训练
    # ------------------------------------------------------------------
    trainer = DQNTrainer(
        env=env,
        observation_builder=observation_builder,
        action_space=action_space,
        agent=agent,
        replay_buffer=replay_buffer,
        trainer_config=trainer_config,
        checkpoint_manager=checkpoint_manager,
        metrics_writer=metrics_writer,
        logger=logger,
    )

    logger.info("Starting DQN training.")
    result = trainer.train()

    # ------------------------------------------------------------------
    # 8. 输出最终训练结果
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("DQN training entry finished.")
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