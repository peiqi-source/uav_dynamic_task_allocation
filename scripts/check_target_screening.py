from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.preprocessing.target_screening import (
    TargetScreener,
    load_target_screening_config,
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

    logger.info("Target screening check started.")

    data = load_all_data(config)

    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    screening_config = load_target_screening_config(config)
    screener = TargetScreener(screening_config)

    screened_set = screener.screen(state.targets)
    screened_set.validate()

    csv_path = screener.write_debug_csv(
        screened_set=screened_set,
        project_root=project_root,
    )

    logger.info("Target screening finished successfully.")
    logger.info(f"All targets: {len(screened_set.all_targets)}")
    logger.info(f"Candidate targets: {len(screened_set.candidate_targets)}")
    logger.info(f"Selected targets: {len(screened_set.selected_targets)}")
    logger.info(f"Debug CSV saved to: {csv_path}")

    top_scores = sorted(
        screened_set.target_scores.values(),
        key=lambda score: score.score,
        reverse=True,
    )[:10]

    logger.info("Top target scores:")
    for rank, target_score in enumerate(top_scores, start=1):
        components = target_score.components
        logger.info(
            f"Rank {rank}: "
            f"target_id={target_score.target_id}, "
            f"score={target_score.score:.6f}, "
            f"raw_score={components.get('raw_score', 0):.6f}, "
            f"adjusted_significance={components.get('adjusted_significance', 0):.6f}, "
            f"adjusted_defense={components.get('adjusted_defense', 0):.6f}, "
            f"normalized_distance={components.get('normalized_distance', 0):.6f}"
        )

    logger.info("Target screening check finished successfully.")


if __name__ == "__main__":
    main()