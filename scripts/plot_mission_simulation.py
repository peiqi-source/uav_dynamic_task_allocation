from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.simulation.mission_visualization import (
    build_mission_visualizer,
)
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

    logger.info("Mission simulation visualization started.")

    data = load_all_data(config)

    battlefield_state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    visualizer = build_mission_visualizer(config)

    figure_paths = visualizer.save_all(
        targets=battlefield_state.targets,
        project_root=project_root,
    )

    logger.info("Mission simulation visualization finished successfully.")

    for path in figure_paths:
        logger.info(f"Figure saved: {path}")


if __name__ == "__main__":
    main()