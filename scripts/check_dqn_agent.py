import torch

from uav_dynamic_task_allocation.algorithms.rl.dqn.agent import (
    DQNAgent,
    load_dqn_agent_config,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.network import (
    load_dqn_network_config,
)
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
from uav_dynamic_task_allocation.utils.device import get_device
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config
from uav_dynamic_task_allocation.utils.seed import set_seed


def main() -> None:
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("DQN agent check started.")

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

    env = DroneBattleEnv(
        uav_df=data["uav"],
        target_df=data["target"],
        env_config=env_config,
        reward_config=reward_config,
    )

    env.reset()

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
        capacity=1000,
        seed=seed,
    )

    logger.info(f"Input dim: {input_dim}")
    logger.info(f"Action dim: {action_dim}")
    logger.info(f"Initial epsilon: {agent.get_epsilon():.6f}")

    collection_steps = agent_config.batch_size

    for step_index in range(collection_steps):
        state = env.get_state_copy()

        observation = observation_builder.build(state)
        action_mask = action_space.build_action_mask(state)

        valid_action_count = int(action_mask.sum())
        if valid_action_count == 0:
            logger.warning(
                "No valid action found. "
                "Please set action_space.require_reachable=false "
                "while testing the DQN pipeline."
            )
            break

        selection = agent.select_action(
            observation=observation.vector,
            action_mask=action_mask,
            training=True,
        )

        decoded_action = action_space.decode_action_id(
            state=state,
            action_id=selection.action_id,
        )

        env_action = decoded_action.to_env_action()
        result = env.step(env_action)

        next_state = env.get_state_copy()
        next_observation = observation_builder.build(next_state)
        next_action_mask = action_space.build_action_mask(next_state)

        transition = Transition(
            state=observation.vector,
            action_id=selection.action_id,
            reward=result.reward,
            next_state=next_observation.vector,
            done=result.done,
            action_mask=action_mask,
            next_action_mask=next_action_mask,
            info={
                "selection": selection.to_dict(),
                "env_info": result.info,
            },
        )

        replay_buffer.push(transition)

        logger.info(
            "Collected transition: "
            f"step={step_index}, "
            f"action_id={selection.action_id}, "
            f"epsilon={selection.epsilon:.4f}, "
            f"is_random={selection.is_random}, "
            f"reward={result.reward:.6f}, "
            f"done={result.done}"
        )

        if result.done:
            logger.info("Environment reached done=True. Resetting environment.")
            env.reset()

    logger.info(f"Replay buffer size: {len(replay_buffer)}")

    if not replay_buffer.can_sample(agent_config.batch_size):
        raise RuntimeError(
            "Replay buffer does not have enough samples for one DQN update. "
            f"required={agent_config.batch_size}, available={len(replay_buffer)}"
        )

    batch = replay_buffer.sample(agent_config.batch_size)

    update_result = agent.update(batch)

    logger.info("DQN update finished.")
    logger.info(f"Update result: {update_result.to_dict()}")
    logger.info(f"Current epsilon after collection: {agent.get_epsilon():.6f}")
    logger.info(f"Total action steps: {agent.total_action_steps}")
    logger.info(f"Total update steps: {agent.total_update_steps}")

    logger.info("DQN agent check finished successfully.")


if __name__ == "__main__":
    main()