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
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("Resource allocation check started.")

    data = load_all_data(config)

    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    # 1. 目标评分
    screening_config = load_target_screening_config(config)
    screener = TargetScreener(screening_config)
    screened_set = screener.screen(state.targets)

    # 2. 摧毁目标集选择
    selector_config = load_destroy_target_selection_config(config)
    selector = DestroyTargetSelector(selector_config)
    selection_result = selector.select(screened_set)
    selected_screened_set = selection_result.to_screened_target_set()

    # 3. 目标分群
    clustering_config = load_target_clustering_config(config)
    clusterer = TargetClusterer(clustering_config)
    clustering_result = clusterer.cluster(selected_screened_set)

    # 4. 构造 UAV 资源池
    resource_status = build_resource_status_from_state(state)

    # 5. 资源分配
    allocation_config = load_resource_allocation_config(config)
    allocator = ResourceAllocator(allocation_config)

    allocation_result = allocator.allocate(
        cluster_set=clustering_result.cluster_set,
        resource_status=resource_status,
    )

    allocation_csv_path = allocator.write_debug_csv(
        result=allocation_result,
        project_root=project_root,
    )

    logger.info("Resource allocation finished successfully.")
    logger.info(f"Destroy targets: {len(selected_screened_set.selected_targets)}")
    logger.info(f"Target clusters: {clustering_result.cluster_set.num_clusters}")
    logger.info(f"Resource allocation CSV: {allocation_csv_path}")

    logger.info(f"Resource status metadata: {resource_status.metadata}")
    logger.info(f"Allocation metadata: {allocation_result.metadata}")

    for assignment in allocation_result.allocation_plan.assignments:
        logger.info(
            f"Cluster {assignment.cluster_id}: "
            f"target_ids={assignment.target_cluster.target_ids}, "
            f"required_attack={assignment.required_attack_uav_count}, "
            f"assigned_attack={assignment.assigned_attack_uav_ids}, "
            f"assigned_guide={assignment.assigned_guide_uav_ids}, "
            f"center=({assignment.target_cluster.center.x:.4f}, "
            f"{assignment.target_cluster.center.y:.4f}), "
            f"shortage={assignment.metadata.get('resource_shortage')}"
        )

    logger.info("Resource allocation check finished successfully.")


if __name__ == "__main__":
    main()