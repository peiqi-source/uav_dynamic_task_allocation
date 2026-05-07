from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.envs.action_space import UAVTargetActionSpace
from uav_dynamic_task_allocation.envs.drone_battle_env import DroneBattleEnv
from uav_dynamic_task_allocation.envs.env_config import load_env_config
from uav_dynamic_task_allocation.utils.config import (
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


def main() -> None:
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("Action space check started.")

    data = load_all_data(config)
    env_config = load_env_config(config)

    env = DroneBattleEnv(
        uav_df=data["uav"],
        target_df=data["target"],
        env_config=env_config,
    )

    observation = env.reset()
    logger.info("Environment reset successfully.")
    logger.info(f"Initial observation: {observation}")

    state = env.get_state_copy()

    # 第一轮：只生成合法动作。
    valid_action_space = UAVTargetActionSpace(
        env_config=env.env_config,
        include_invalid_actions=False,
    )
    valid_actions = valid_action_space.build(state)

    logger.info("Valid action space built successfully.")
    logger.info(f"Valid action space summary: {valid_action_space.summary()}")

    if valid_actions:
        first_action = valid_actions[0]
        logger.info(f"First valid action: {first_action.to_dict()}")

        env_action = first_action.to_env_action()
        logger.info(f"Converted env action: {env_action}")

        result = env.step(env_action)
        logger.info("Executed first valid action in environment.")
        logger.info(f"Reward: {result.reward}")
        logger.info(f"Done: {result.done}")
        logger.info(f"Info: {result.info}")
    else:
        logger.warning(
            "No valid action found. This may happen if all attack UAVs "
            "are out of range or there is no active target."
        )

    # 第二轮：生成所有候选动作，包括非法动作，便于查看过滤原因。
    debug_action_space = UAVTargetActionSpace(
        env_config=env.env_config,
        include_invalid_actions=True,
    )
    debug_action_space.build(state)

    logger.info("Debug action space built successfully.")
    logger.info(f"Debug action space summary: {debug_action_space.summary()}")

    invalid_actions = debug_action_space.invalid_actions()
    if invalid_actions:
        logger.info("First 5 invalid actions:")
        for action in invalid_actions[:5]:
            logger.info(action.to_dict())

    logger.info("Action space check finished successfully.")


if __name__ == "__main__":
    main()