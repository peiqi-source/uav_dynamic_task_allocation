from uav_dynamic_task_allocation.algorithms.rl.dqn.replay_buffer import (
    ReplayBuffer,
    Transition,
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
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config
from uav_dynamic_task_allocation.utils.seed import set_seed


def main() -> None:
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("Replay buffer check started.")

    seed = int(get_config_value(config, "experiment.seed", default=42))
    set_seed(seed)

    data = load_all_data(config)
    env_config = load_env_config(config)
    reward_config = load_reward_config(config)
    obs_config = load_observation_config(config)
    action_space_config = load_action_space_config(config)

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

    replay_buffer = ReplayBuffer(
        capacity=1000,
        seed=seed,
    )

    env.reset()

    max_collection_steps = 10

    for step_index in range(max_collection_steps):
        state = env.get_state_copy()

        observation = observation_builder.build(state)
        action_mask = action_space.build_action_mask(state)

        valid_action_ids = action_space.get_valid_action_ids(state)

        if not valid_action_ids:
            logger.warning(
                "No valid action found while collecting replay buffer samples. "
                "You may temporarily set action_space.require_reachable=false "
                "in configs/default.yaml."
            )
            break

        # 当前测试脚本使用第一个合法动作。
        # 后续 DQNAgent 会使用 epsilon-greedy 策略选择动作。
        action_id = valid_action_ids[0]
        decoded_action = action_space.decode_action_id(state, action_id)
        env_action = decoded_action.to_env_action()

        result = env.step(env_action)

        next_state = env.get_state_copy()
        next_observation = observation_builder.build(next_state)
        next_action_mask = action_space.build_action_mask(next_state)

        transition = Transition(
            state=observation.vector,
            action_id=action_id,
            reward=result.reward,
            next_state=next_observation.vector,
            done=result.done,
            action_mask=action_mask,
            next_action_mask=next_action_mask,
            info=result.info,
        )

        replay_buffer.push(transition)

        logger.info(
            "Transition pushed: "
            f"step={step_index}, "
            f"action_id={action_id}, "
            f"reward={result.reward:.6f}, "
            f"done={result.done}"
        )

        if result.done:
            logger.info("Environment reached done=True during sample collection.")
            break

    logger.info(f"Replay buffer size: {len(replay_buffer)}")

    if not replay_buffer.can_sample(batch_size=1):
        raise RuntimeError(
            "Replay buffer has no sample. "
            "Please check action_space configuration and valid action count."
        )

    sample_batch_size = min(4, len(replay_buffer))
    batch = replay_buffer.sample(sample_batch_size)

    logger.info("Sampled batch successfully.")
    logger.info(f"Batch size: {batch.batch_size}")
    logger.info(f"states shape: {batch.states.shape}")
    logger.info(f"action_ids shape: {batch.action_ids.shape}")
    logger.info(f"rewards shape: {batch.rewards.shape}")
    logger.info(f"next_states shape: {batch.next_states.shape}")
    logger.info(f"dones shape: {batch.dones.shape}")

    if batch.action_masks is not None:
        logger.info(f"action_masks shape: {batch.action_masks.shape}")

    if batch.next_action_masks is not None:
        logger.info(f"next_action_masks shape: {batch.next_action_masks.shape}")

    logger.info("First sampled transition info:")
    logger.info(batch.infos[0])

    logger.info("Replay buffer check finished successfully.")


if __name__ == "__main__":
    main()