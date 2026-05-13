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
from uav_dynamic_task_allocation.planning.strike_order_planner import (
    build_strike_order_planner,
)
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
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("StrikeOrderPlanner check started.")

    data = load_all_data(config)

    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    # 1. 目标评分
    screener = TargetScreener(load_target_screening_config(config))
    screened_set = screener.screen(state.targets)

    # 2. 摧毁目标集选择
    selector = DestroyTargetSelector(load_destroy_target_selection_config(config))
    selection_result = selector.select(screened_set)
    selected_screened_set = selection_result.to_screened_target_set()

    # 3. 目标分群
    clusterer = TargetClusterer(load_target_clustering_config(config))
    clustering_result = clusterer.cluster(selected_screened_set)

    # 4. 资源分配
    resource_status = build_resource_status_from_state(state)
    allocator = ResourceAllocator(load_resource_allocation_config(config))
    allocation_result = allocator.allocate(
        cluster_set=clustering_result.cluster_set,
        resource_status=resource_status,
    )

    # 5. 群内打击次序规划
    strike_order_planner = build_strike_order_planner(
        config=config,
        logger=logger,
    )

    strike_order_result = strike_order_planner.plan(
        target_clusters=clustering_result.cluster_set,
        allocation_plan=allocation_result.allocation_plan,
    )

    csv_path = strike_order_planner.write_debug_csv(
        result=strike_order_result,
        project_root=project_root,
    )

    logger.info("StrikeOrderPlanner finished successfully.")
    logger.info(f"Strike order CSV: {csv_path}")
    logger.info(f"Metadata: {strike_order_result.metadata}")

    for cluster_id, plan in strike_order_result.strike_order_plans.items():
        logger.info(
            f"Cluster {cluster_id}: "
            f"ordered_target_ids={plan.ordered_target_ids}, "
            f"total_path_distance={plan.total_path_distance}, "
            f"expected_reward={plan.expected_reward}"
        )

    logger.info("StrikeOrderPlanner check finished successfully.")


if __name__ == "__main__":
    main()