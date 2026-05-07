import torch

from uav_dynamic_task_allocation.algorithms.rl.dqn.network import (
    DQNNetwork,
    apply_action_mask,
    load_dqn_network_config,
    select_greedy_action,
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

    logger.info("DQN network check started.")

    seed = int(get_config_value(config, "experiment.seed", default=42))
    preferred_device = str(
        get_config_value(config, "experiment.device", default="auto")
    )

    set_seed(seed)

    device_name = get_device(preferred_device)
    device = torch.device(device_name)

    logger.info(f"Selected device: {device}")

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

    env.reset()
    state = env.get_state_copy()

    observation_builder = ObservationBuilder(
        env_config=env.env_config,
        obs_config=obs_config,
    )

    action_space = FixedUAVTargetActionSpace(
        env_config=env.env_config,
        action_space_config=action_space_config,
    )

    observation = observation_builder.build(state)
    action_mask = action_space.build_action_mask(state)

    input_dim = observation_builder.get_input_dim()
    action_dim = action_space.size

    network_config = load_dqn_network_config(
        config=config,
        input_dim=input_dim,
        action_dim=action_dim,
    )

    logger.info(f"Input dim: {input_dim}")
    logger.info(f"Action dim: {action_dim}")
    logger.info(f"Hidden dims: {network_config.hidden_dims}")
    logger.info(f"Activation: {network_config.activation}")

    q_network = DQNNetwork(network_config).to(device)
    q_network.eval()

    observation_tensor = torch.from_numpy(observation.vector).float().unsqueeze(0)
    observation_tensor = observation_tensor.to(device)

    action_mask_tensor = torch.from_numpy(action_mask).float().unsqueeze(0)
    action_mask_tensor = action_mask_tensor.to(device)

    with torch.no_grad():
        q_values = q_network(observation_tensor)

    logger.info(f"Q values shape: {q_values.shape}")
    logger.info(
        "Q values sample: "
        f"min={q_values.min().item():.6f}, "
        f"max={q_values.max().item():.6f}, "
        f"mean={q_values.mean().item():.6f}"
    )

    valid_action_count = int(action_mask.sum())
    logger.info(f"Valid action count: {valid_action_count}")

    if valid_action_count > 0:
        with torch.no_grad():
            masked_q_values = apply_action_mask(q_values, action_mask_tensor)
            selected_action_tensor = select_greedy_action(
                q_values,
                action_mask_tensor,
            )

        selected_action_id = int(selected_action_tensor.item())
        decoded_action = action_space.decode_action_id(
            state=state,
            action_id=selected_action_id,
        )

        logger.info(f"Selected action_id: {selected_action_id}")
        logger.info(f"Decoded selected action: {decoded_action.to_dict()}")

        env_action = decoded_action.to_env_action()
        result = env.step(env_action)

        logger.info("Executed selected action in environment.")
        logger.info(f"Reward: {result.reward}")
        logger.info(f"Done: {result.done}")
        logger.info(f"Info: {result.info}")
        logger.info(
            "Masked Q values sample: "
            f"min={masked_q_values.min().item():.6f}, "
            f"max={masked_q_values.max().item():.6f}, "
            f"mean={masked_q_values.mean().item():.6f}"
        )
    else:
        logger.warning(
            "No valid action found. Network forward pass is correct, "
            "but masked action selection is skipped. "
            "You may temporarily set action_space.require_reachable=false."
        )

    logger.info("DQN network check finished successfully.")


if __name__ == "__main__":
    main()