"""checkdestroy目标selection脚本，封装可直接运行的实验、检查或可视化流程。"""
from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.preprocessing.destroy_target_selection import (
    DestroyTargetSelector,
    load_destroy_target_selection_config,
)
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

    logger.info("Destroy target selection check started.")

    data = load_all_data(config)

    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    screening_config = load_target_screening_config(config)
    screener = TargetScreener(screening_config)

    screened_set = screener.screen(state.targets)
    screening_csv_path = screener.write_debug_csv(
        screened_set=screened_set,
        project_root=project_root,
    )

    selector_config = load_destroy_target_selection_config(config)
    selector = DestroyTargetSelector(selector_config)

    result = selector.select(screened_set)
    selected_screened_set = result.to_screened_target_set()

    selection_csv_path = selector.write_debug_csv(
        result=result,
        project_root=project_root,
    )

    logger.info("Destroy target selection finished successfully.")
    logger.info(f"Target screening CSV: {screening_csv_path}")
    logger.info(f"Destroy target selection CSV: {selection_csv_path}")

    logger.info(f"All targets: {len(selected_screened_set.all_targets)}")
    logger.info(f"Candidate targets: {len(selected_screened_set.candidate_targets)}")
    logger.info(f"Destroy targets: {len(result.destroy_targets)}")
    logger.info(f"Non-destroy targets: {len(result.non_destroy_targets)}")
    logger.info(f"Selected cluster label: {result.selected_cluster_label}")
    logger.info(f"Metadata: {result.metadata}")

    destroy_ids = [target.target_id for target in result.destroy_targets]
    non_destroy_ids = [target.target_id for target in result.non_destroy_targets]

    logger.info(f"Destroy target ids: {destroy_ids}")
    logger.info(f"Non-destroy target ids: {non_destroy_ids}")

    logger.info("Destroy target selection check finished successfully.")


if __name__ == "__main__":
    main()