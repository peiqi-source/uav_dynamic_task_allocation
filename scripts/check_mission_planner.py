from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.planning.algorithm_policy import (
    load_algorithm_policy,
)
from uav_dynamic_task_allocation.planning.mission_planner import (
    MissionPlanner,
    load_mission_planner_config,
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

    logger.info("Mission planner check started.")

    data = load_all_data(config)

    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    algorithm_policy = load_algorithm_policy(config)
    planner_config = load_mission_planner_config(config)

    planner = MissionPlanner(
        base_config=config,
        algorithm_policy=algorithm_policy,
        planner_config=planner_config,
        logger=logger,
    )

    result = planner.plan(
        battlefield_state=state,
    )

    csv_path = planner.write_debug_csv(
        result=result,
        project_root=project_root,
    )

    logger.info("Mission planner finished successfully.")
    logger.info(f"Planner summary CSV: {csv_path}")
    logger.info(f"Planner metadata: {result.metadata}")

    logger.info(
        "Destroy targets: "
        f"{[target.target_id for target in result.selected_screened_set.selected_targets]}"
    )

    for cluster in result.target_clustering_result.cluster_set.clusters:
        logger.info(
            f"Cluster {cluster.cluster_id}: "
            f"target_ids={cluster.target_ids}, "
            f"center=({cluster.center.x:.4f}, {cluster.center.y:.4f}), "
            f"defense_sum={cluster.defense_sum:.4f}, "
            f"significance_sum={cluster.significance_sum:.4f}"
        )

    for assignment in result.resource_allocation_result.allocation_plan.assignments:
        logger.info(
            f"Assignment cluster={assignment.cluster_id}: "
            f"targets={assignment.target_cluster.target_ids}, "
            f"required_attack={assignment.required_attack_uav_count}, "
            f"assigned_attack={assignment.assigned_attack_uav_ids}, "
            f"assigned_guide={assignment.assigned_guide_uav_ids}, "
            f"shortage={assignment.metadata.get('resource_shortage')}"
        )

    for cluster_id, strike_plan in result.mission_plan.strike_order_plans.items():
        logger.info(
            f"Strike order cluster={cluster_id}: "
            f"ordered_target_ids={strike_plan.ordered_target_ids}, "
            f"total_path_distance={strike_plan.total_path_distance}, "
            f"expected_reward={strike_plan.expected_reward}"
        )

    logger.info("Mission planner check finished successfully.")


if __name__ == "__main__":
    main()