"""checkDQN 算法检查点脚本，封装可直接运行的实验、检查或可视化流程。"""
import torch

from uav_dynamic_task_allocation.algorithms.rl.dqn.agent import (
    DQNAgent,
    load_dqn_agent_config,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.checkpoint import (
    DQNCheckpointManager,
    DQNCheckpointMetadata,
    load_dqn_checkpoint_config,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.network import (
    load_dqn_network_config,
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


def parameters_are_equal(agent_a: DQNAgent, agent_b: DQNAgent) -> bool:
    """
    检查两个 agent 的 policy network 参数是否完全一致。

    checkpoint 加载后，如果参数一致，说明保存和恢复过程是正确的。
    """
    for param_a, param_b in zip(
        agent_a.policy_network.parameters(),
        agent_b.policy_network.parameters(),
        strict=False,
    ):
        if not torch.equal(param_a.detach().cpu(), param_b.detach().cpu()):
            return False

    return True


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

    logger.info("DQN checkpoint check started.")

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
    checkpoint_config = load_dqn_checkpoint_config(config)

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

    checkpoint_manager = DQNCheckpointManager(
        config=checkpoint_config,
        project_root=project_root,
    )

    metadata = DQNCheckpointMetadata(
        episode=1,
        global_env_steps=10,
        total_action_steps=agent.total_action_steps,
        total_update_steps=agent.total_update_steps,
        best_reward=12.5,
        last_reward=10.0,
        epsilon=agent.get_epsilon(),
        extra={
            "check_script": "check_dqn_checkpoint.py",
            "input_dim": input_dim,
            "action_dim": action_dim,
        },
    )

    latest_path = checkpoint_manager.save_latest(
        agent=agent,
        metadata=metadata,
    )

    logger.info(f"Latest checkpoint saved successfully: {latest_path}")

    best_path = checkpoint_manager.save_best(
        agent=agent,
        metadata=metadata,
    )

    logger.info(f"Best checkpoint saved successfully: {best_path}")

    # 创建一个新的 agent，用于测试是否能从 checkpoint 正确恢复参数。
    restored_agent = DQNAgent(
        network_config=network_config,
        agent_config=agent_config,
        device=device,
        seed=seed + 1,
    )

    load_result = checkpoint_manager.load_latest(
        agent=restored_agent,
        load_optimizer=True,
        map_location=device,
    )

    logger.info(f"Checkpoint loaded successfully from: {load_result.path}")
    logger.info(f"Loaded metadata: {load_result.metadata}")

    if not parameters_are_equal(agent, restored_agent):
        raise RuntimeError(
            "Parameter check failed: restored agent parameters "
            "do not match original agent."
        )

    logger.info("Parameter check passed.")

    logger.info("DQN checkpoint check finished successfully.")


if __name__ == "__main__":
    main()