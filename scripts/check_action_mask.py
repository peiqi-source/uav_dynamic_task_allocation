from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.envs.action_space import (
    FixedUAVTargetActionSpace,
    load_action_space_config,
)
from uav_dynamic_task_allocation.envs.drone_battle_env import DroneBattleEnv
from uav_dynamic_task_allocation.envs.env_config import load_env_config
from uav_dynamic_task_allocation.envs.reward import load_reward_config
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

    logger.info("Fixed action mask check started.")

    data = load_all_data(config)
    env_config = load_env_config(config)
    reward_config = load_reward_config(config)
    action_space_config = load_action_space_config(config)

    env = DroneBattleEnv(
        uav_df=data["uav"],
        target_df=data["target"],
        env_config=env_config,
        reward_config=reward_config,
    )

    env.reset()
    state = env.get_state_copy()

    fixed_action_space = FixedUAVTargetActionSpace(
        env_config=env.env_config,
        action_space_config=action_space_config,
    )

    mask = fixed_action_space.build_action_mask(state)
    summary = fixed_action_space.summarize_mask(state)

    logger.info("Fixed action space built successfully.")
    logger.info(f"Fixed action space size: {fixed_action_space.size}")
    logger.info(f"Action mask shape: {mask.shape}")
    logger.info(f"Mask summary: {summary}")

    logger.info("Testing action_id encoding and decoding.")

    action_id = fixed_action_space.encode_action_id(
        uav_slot=0,
        target_slot=0,
    )
    decoded = fixed_action_space.decode_action_id(state, action_id)

    logger.info(f"Encoded action_id for slot (0, 0): {action_id}")
    logger.info(f"Decoded action: {decoded.to_dict()}")

    valid_action_ids = fixed_action_space.get_valid_action_ids(state)

    if valid_action_ids:
        selected = fixed_action_space.select_first_valid_action(state)

        logger.info("Selected first valid action.")
        logger.info(f"Selected decoded action: {selected.to_dict()}")

        env_action = selected.to_env_action()
        logger.info(f"Converted env action: {env_action}")

        result = env.step(env_action)

        logger.info("Executed selected valid action in environment.")
        logger.info(f"Reward: {result.reward}")
        logger.info(f"Done: {result.done}")
        logger.info(f"Info: {result.info}")
        logger.info(f"Reward breakdown: {result.info.get('reward_breakdown')}")

        next_state = env.get_state_copy()
        next_mask = fixed_action_space.build_action_mask(next_state)
        next_summary = fixed_action_space.summarize_mask(next_state)

        logger.info("Action mask after one environment step:")
        logger.info(f"Next mask valid count: {int(next_mask.sum())}")
        logger.info(f"Next mask summary: {next_summary}")

    else:
        logger.warning("No valid action found in current state.")
        logger.warning(
            "This may happen if all attack UAVs are out of range, "
            "all targets are inactive, or action_space constraints are too strict."
        )

    logger.info("Fixed action mask check finished successfully.")


if __name__ == "__main__":
    main()