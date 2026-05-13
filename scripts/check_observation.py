"""check观测向量脚本，封装可直接运行的实验、检查或可视化流程。"""
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.envs.drone_battle_env import DroneBattleEnv
from uav_dynamic_task_allocation.envs.env_config import load_env_config
from uav_dynamic_task_allocation.envs.observation import (
    ObservationBuilder,
    load_observation_config,
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

    logger.info("Observation check started.")

    data = load_all_data(config)
    env_config = load_env_config(config)
    obs_config = load_observation_config(config)

    env = DroneBattleEnv(
        uav_df=data["uav"],
        target_df=data["target"],
        env_config=env_config,
    )
    env.reset()

    state = env.get_state_copy()

    observation_builder = ObservationBuilder(
        env_config=env.env_config,
        obs_config=obs_config,
    )

    observation = observation_builder.build(state)

    logger.info("Observation built successfully.")
    logger.info(f"Observation vector shape: {observation.vector.shape}")
    logger.info(f"Observation vector dtype: {observation.vector.dtype}")
    logger.info(f"Expected input dim: {observation_builder.get_input_dim()}")
    logger.info(f"Debug info: {observation.debug_info}")

    logger.info(
        "UAV mask summary: "
        f"shape={observation.masks['uav_mask'].shape}, "
        f"valid={observation.masks['uav_mask'].sum()}"
    )
    logger.info(
        "Target mask summary: "
        f"shape={observation.masks['target_mask'].shape}, "
        f"valid={observation.masks['target_mask'].sum()}"
    )

    if observation.feature_names:
        logger.info("First 20 observation features:")
        for name, value in zip(
            observation.feature_names[:20],
            observation.vector[:20],
            strict=False,
        ):
            logger.info(f"  {name}: {value:.6f}")

    if observation.vector.shape[0] != observation_builder.get_input_dim():
        raise RuntimeError(
            "Observation dimension mismatch: "
            f"actual={observation.vector.shape[0]}, "
            f"expected={observation_builder.get_input_dim()}"
        )

    logger.info("Observation check finished successfully.")


if __name__ == "__main__":
    main()