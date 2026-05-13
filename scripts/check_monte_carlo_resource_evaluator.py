"""checkmontecarlo资源evaluator脚本，封装可直接运行的实验、检查或可视化流程。"""
from uav_dynamic_task_allocation.allocation.monte_carlo_resource_evaluator import (
    MonteCarloResourceEvaluator,
    load_monte_carlo_resource_evaluator_config,
)
from uav_dynamic_task_allocation.allocation.resource_allocation import (
    ResourceAllocator,
    build_resource_status_from_state,
    load_resource_allocation_config,
)
from uav_dynamic_task_allocation.allocation.target_clustering import (
    TargetClusterer,
    load_target_clustering_config,
)
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
    config = load_and_validate_config(project_root / "configs" / "default.yaml")
    logger = setup_logger_from_config(config)

    logger.info("Monte Carlo resource evaluator check started.")

    data = load_all_data(config)
    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    screened_set = TargetScreener(
        load_target_screening_config(config)
    ).screen(state.targets)
    selected_screened_set = DestroyTargetSelector(
        load_destroy_target_selection_config(config)
    ).select(screened_set).to_screened_target_set()
    cluster_set = TargetClusterer(
        load_target_clustering_config(config)
    ).cluster(selected_screened_set).cluster_set

    allocation_result = ResourceAllocator(
        load_resource_allocation_config(config)
    ).allocate(
        cluster_set=cluster_set,
        resource_status=build_resource_status_from_state(state),
    )

    evaluator = MonteCarloResourceEvaluator(
        load_monte_carlo_resource_evaluator_config(config)
    )
    results = evaluator.assess_allocation(allocation_result.allocation_plan)
    csv_path = evaluator.write_csv(results, project_root=project_root)
    samples_csv_path = evaluator.write_samples_csv(results, project_root=project_root)
    figure_path = evaluator.write_distribution_plot(results, project_root=project_root)

    logger.info(f"Monte Carlo CSV: {csv_path}")
    logger.info(f"Monte Carlo samples CSV: {samples_csv_path}")
    logger.info(f"Monte Carlo figure: {figure_path}")
    for result in results:
        logger.info(result.to_dict())

    logger.info("Monte Carlo resource evaluator check finished successfully.")


if __name__ == "__main__":
    main()
