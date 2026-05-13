"""checkDQN 算法训练器脚本，封装可直接运行的实验、检查或可视化流程。"""
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


def main() -> None:
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        无返回值；通过状态变更、文件输出或日志记录体现执行结果。
    """
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("DQN trainer check started.")

    seed = int(get_config_value(config, "experiment.seed", default=42))
    preferred_device = str(
        get_config_value(config, "experiment.device", default="auto")
    )

    set_seed(seed)

    device = torch.device(get_device(preferred_device))
    logger.info(f"Selected device: {device}")

    data = load_all_data(config)

    env_config = load_env_config(config)
    reward_config = load_reward_config(config)
    obs_config = load_observation_config(config)
    action_space_config = load_action_space_config(config)
    agent_config = load_dqn_agent_config(config)
    trainer_config = load_dqn_trainer_config(config)
    checkpoint_config = load_dqn_checkpoint_config(config)
    metrics_config = load_dqn_metrics_config(config)

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

    logger.info(f"Input dim: {input_dim}")
    logger.info(f"Action dim: {action_dim}")
    logger.info(f"Trainer config: {trainer_config}")

    result = trainer.train()

    logger.info("Trainer returned result successfully.")
    logger.info(f"Best episode: {result.best_episode}")
    logger.info(f"Best reward: {result.best_reward}")
    logger.info(f"Last reward: {result.last_reward}")

    logger.info("Episode metrics:")
    for metric in result.episode_metrics:
        logger.info(metric.to_dict())

    if checkpoint_manager is not None:
        logger.info(f"Latest checkpoint path: {checkpoint_manager.latest_path}")
        logger.info(f"Best checkpoint path: {checkpoint_manager.best_path}")

        if not checkpoint_manager.latest_path.exists():
            raise RuntimeError(
                f"Latest checkpoint was not saved: {checkpoint_manager.latest_path}"
            )

        if not checkpoint_manager.best_path.exists():
            raise RuntimeError(
                f"Best checkpoint was not saved: {checkpoint_manager.best_path}"
            )

        logger.info("Checkpoint files verified successfully.")

    if metrics_writer is not None:
        logger.info(f"Metrics CSV path: {metrics_writer.csv_path}")

        if not metrics_writer.csv_path.exists():
            raise RuntimeError(
                f"Metrics CSV was not saved: {metrics_writer.csv_path}"
            )

        if metrics_config.save_jsonl and not metrics_writer.jsonl_path.exists():
            raise RuntimeError(
                f"Metrics JSONL was not saved: {metrics_writer.jsonl_path}"
            )

        logger.info("Metrics files verified successfully.")

    logger.info("DQN trainer check finished successfully.")


if __name__ == "__main__":
    main()