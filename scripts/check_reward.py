"""check奖励脚本，封装可直接运行的实验、检查或可视化流程。"""
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.envs.drone_battle_env import DroneBattleEnv
from uav_dynamic_task_allocation.envs.env_config import load_env_config
from uav_dynamic_task_allocation.envs.reward import (
    RewardCalculator,
    RewardContext,
    load_reward_config,
)
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

    logger.info("Reward check started.")

    data = load_all_data(config)
    env_config = load_env_config(config)
    reward_config = load_reward_config(config)

    env = DroneBattleEnv(
        uav_df=data["uav"],
        target_df=data["target"],
        env_config=env_config,
    )
    env.reset()

    state = env.get_state_copy()

    if not state.attack_uavs:
        logger.warning("No attack UAV found. Cannot test valid reward.")
        return

    if not state.active_targets:
        logger.warning("No active target found. Cannot test valid reward.")
        return

    reward_calculator = RewardCalculator(
        env_config=env.env_config,
        reward_config=reward_config,
    )

    test_uav = state.attack_uavs[0]
    test_target = state.active_targets[0]

    logger.info(
        "Testing valid reward with "
        f"uav_id={test_uav.uav_id}, target_id={test_target.target_id}"
    )

    valid_context = RewardContext(
        uav=test_uav,
        target=test_target,
        current_step=max(state.current_step + 1, 1),
        current_time=(state.current_step + 1) * env.env_config.time_step,
        is_valid_action=True,
        target_was_active_before_action=True,
        target_destroyed_after_action=False,
        target_damaged_after_action=False,
        previous_target_defense=test_target.defense,
        remaining_target_defense=test_target.defense,
        attacked_target_ids_before_action=set(),
    )

    valid_reward = reward_calculator.calculate(valid_context)

    logger.info("Valid reward breakdown:")
    logger.info(valid_reward.to_dict())

    logger.info("Testing repeated target reward.")

    repeated_context = RewardContext(
        uav=test_uav,
        target=test_target,
        current_step=max(state.current_step + 2, 1),
        current_time=(state.current_step + 2) * env.env_config.time_step,
        is_valid_action=True,
        target_was_active_before_action=True,
        attacked_target_ids_before_action={test_target.target_id},
    )

    repeated_reward = reward_calculator.calculate(repeated_context)

    logger.info("Repeated target reward breakdown:")
    logger.info(repeated_reward.to_dict())

    logger.info("Testing invalid action reward.")

    invalid_context = RewardContext(
        uav=test_uav,
        target=test_target,
        current_step=max(state.current_step + 3, 1),
        current_time=(state.current_step + 3) * env.env_config.time_step,
        is_valid_action=False,
        invalid_reason="out_of_range",
    )

    invalid_reward = reward_calculator.calculate(invalid_context)

    logger.info("Invalid action reward breakdown:")
    logger.info(invalid_reward.to_dict())

    logger.info("Reward check finished successfully.")


if __name__ == "__main__":
    main()