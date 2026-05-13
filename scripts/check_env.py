"""check环境脚本，封装可直接运行的实验、检查或可视化流程。"""
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.envs.drone_battle_env import DroneBattleEnv
from uav_dynamic_task_allocation.envs.env_config import load_env_config
from uav_dynamic_task_allocation.envs.reward import load_reward_config
from uav_dynamic_task_allocation.utils.config import (
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


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

    logger.info("Environment check started.")

    data = load_all_data(config)
    env_config = load_env_config(config)
    reward_config = load_reward_config(config)

    logger.info("Environment config loaded successfully.")
    logger.info(f"Coordinate mode: {env_config.coordinate_mode}")
    logger.info(f"Configured max time: {env_config.max_time}")
    logger.info(f"Configured time step: {env_config.time_step}")
    logger.info(f"Computed max steps: {env_config.max_steps}")
    logger.info(
        "Configured base position: "
        f"({env_config.base_position.x}, {env_config.base_position.y})"
    )

    logger.info("Reward config loaded successfully.")
    logger.info(f"Reward mode: {reward_config.mode}")
    logger.info(f"Use normalized distance: {reward_config.use_normalized_distance}")
    logger.info(f"Repetition penalty: {reward_config.repetition_penalty}")

    env = DroneBattleEnv(
        uav_df=data["uav"],
        target_df=data["target"],
        env_config=env_config,
        reward_config=reward_config,
    )

    observation = env.reset()
    logger.info("Environment reset successfully.")
    logger.info(f"Initial observation: {observation}")

    state = env.get_state_copy()

    if not state.attack_uavs:
        logger.warning("No attack UAV found. Cannot test attack action.")
        return

    if not state.active_targets:
        logger.warning("No active target found. Cannot test attack action.")
        return

    test_uav = state.attack_uavs[0]
    test_target = state.active_targets[0]

    action = {
        "uav_id": test_uav.uav_id,
        "target_id": test_target.target_id,
    }

    logger.info(f"Test action: {action}")

    result = env.step(action)

    logger.info("Environment step executed.")
    logger.info(f"Reward: {result.reward}")
    logger.info(f"Done: {result.done}")
    logger.info(f"Info: {result.info}")
    logger.info(f"Reward breakdown: {result.info.get('reward_breakdown')}")
    logger.info(f"Next observation: {result.observation}")

    logger.info("Testing repeated action with the same target.")

    repeated_result = env.step(action)

    logger.info("Repeated action executed.")
    logger.info(f"Repeated reward: {repeated_result.reward}")
    logger.info(f"Repeated done: {repeated_result.done}")
    logger.info(f"Repeated info: {repeated_result.info}")
    logger.info(
        "Repeated reward breakdown: "
        f"{repeated_result.info.get('reward_breakdown')}"
    )

    logger.info("Environment check finished successfully.")


if __name__ == "__main__":
    main()