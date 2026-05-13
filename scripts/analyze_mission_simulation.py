from uav_dynamic_task_allocation.simulation.mission_metrics import (
    build_mission_metrics_analyzer,
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

    logger.info("Mission simulation metrics analysis started.")

    analyzer = build_mission_metrics_analyzer(config)

    result = analyzer.analyze(project_root=project_root)

    summary_path, timeline_path = analyzer.write_outputs(
        result=result,
        project_root=project_root,
    )

    figure_paths = analyzer.save_plots(
        result=result,
        project_root=project_root,
    )

    logger.info("Mission simulation metrics analysis finished successfully.")
    logger.info(f"Summary CSV: {summary_path}")
    logger.info(f"Timeline metrics CSV: {timeline_path}")
    logger.info(f"Figure paths: {figure_paths}")
    logger.info(f"Summary: {result.summary}")


if __name__ == "__main__":
    main()