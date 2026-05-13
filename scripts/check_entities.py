"""checkentities脚本，封装可直接运行的实验、检查或可视化流程。"""
from uav_dynamic_task_allocation.core.entities import (
    BattlefieldState,
    build_battlefield_state,
)
from uav_dynamic_task_allocation.data.loaders import load_all_data
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

    logger.info("Entity check started.")

    data = load_all_data(config)

    state: BattlefieldState = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    logger.info("Battlefield state built successfully.")
    logger.info(f"State summary: {state.summary()}")

    if state.uavs:
        first_uav = state.uavs[0]
        logger.info(
            "First UAV: "
            f"id={first_uav.uav_id}, "
            f"type={first_uav.uav_type.name}, "
            f"position=({first_uav.position.x}, {first_uav.position.y}), "
            f"work_range={first_uav.work_range}, "
            f"attack_power={first_uav.attack_power}"
        )

    if state.targets:
        first_target = state.targets[0]
        logger.info(
            "First Target: "
            f"id={first_target.target_id}, "
            f"type={first_target.target_type}, "
            f"position=({first_target.position.x}, {first_target.position.y}), "
            f"defense={first_target.defense}, "
            f"significance={first_target.significance}"
        )

    if state.uavs and state.targets:
        distance = state.uavs[0].distance_to_target(state.targets[0])
        reachable = state.uavs[0].can_reach(state.targets[0])

        logger.info(
            "Distance from first UAV to first target: "
            f"{distance:.2f}"
        )
        logger.info(
            "Can first UAV reach first target: "
            f"{reachable}"
        )

    logger.info("Entity check finished successfully.")


if __name__ == "__main__":
    main()