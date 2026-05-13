"""checkcontracts脚本，封装可直接运行的实验、检查或可视化流程。"""
from uav_dynamic_task_allocation.core.contracts import (
    AlgorithmMetadata,
    AllocationPlan,
    ClusterAssignment,
    MissionEvent,
    MissionEventType,
    MissionMetrics,
    MissionPlan,
    MissionResult,
    MissionTimeSnapshot,
    PipelineStage,
    ResourceStatus,
    ScreenedTargetSet,
    StrikeOrderPlan,
    TargetCluster,
    TargetClusterSet,
)
from uav_dynamic_task_allocation.core.entities import Position, build_battlefield_state
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

    logger.info("Contracts check started.")

    data = load_all_data(config)

    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    targets = state.targets
    uavs = state.uavs

    if len(targets) < 2:
        raise RuntimeError("Need at least 2 targets to check contracts.")

    if len(uavs) < 2:
        raise RuntimeError("Need at least 2 UAVs to check contracts.")

    selected_targets = targets[: min(4, len(targets))]

    screening_metadata = AlgorithmMetadata(
        algorithm_name="simple_top_k_screening",
        algorithm_type="rule_based",
        stage=PipelineStage.TARGET_SCREENING,
        config={"top_k": len(selected_targets)},
        notes="This is a contract check placeholder, not the final screening algorithm.",
    )

    screened_set = ScreenedTargetSet(
        all_targets=targets,
        candidate_targets=selected_targets,
        selected_targets=selected_targets,
        algorithm_metadata=screening_metadata,
        metadata={"source": "check_contracts.py"},
    )
    screened_set.validate()

    first_cluster_targets = selected_targets[:2]
    second_cluster_targets = selected_targets[2:]

    if not second_cluster_targets:
        second_cluster_targets = selected_targets[:1]

    cluster_0 = TargetCluster(
        cluster_id=0,
        targets=first_cluster_targets,
        center=Position(
            x=sum(target.position.x for target in first_cluster_targets)
            / len(first_cluster_targets),
            y=sum(target.position.y for target in first_cluster_targets)
            / len(first_cluster_targets),
        ),
        defense_sum=sum(target.defense for target in first_cluster_targets),
        significance_sum=sum(target.significance for target in first_cluster_targets),
        compactness=None,
        metadata={"method": "manual_contract_check"},
    )

    cluster_1 = TargetCluster(
        cluster_id=1,
        targets=second_cluster_targets,
        center=Position(
            x=sum(target.position.x for target in second_cluster_targets)
            / len(second_cluster_targets),
            y=sum(target.position.y for target in second_cluster_targets)
            / len(second_cluster_targets),
        ),
        defense_sum=sum(target.defense for target in second_cluster_targets),
        significance_sum=sum(target.significance for target in second_cluster_targets),
        compactness=None,
        metadata={"method": "manual_contract_check"},
    )

    cluster_set = TargetClusterSet(
        clusters=[cluster_0, cluster_1],
        algorithm_metadata=AlgorithmMetadata(
            algorithm_name="manual_two_cluster",
            algorithm_type="debug",
            stage=PipelineStage.TARGET_CLUSTERING,
        ),
    )
    cluster_set.validate()

    attack_uavs = state.attack_uavs
    guide_uavs = [
        uav for uav in state.uavs
        if uav not in attack_uavs
    ]

    resource_status = ResourceStatus(
        all_uavs=uavs,
        available_attack_uavs=attack_uavs,
        available_guide_uavs=guide_uavs,
    )

    assignment_0 = ClusterAssignment(
        cluster_id=0,
        target_cluster=cluster_0,
        assigned_attack_uavs=attack_uavs[:2],
        assigned_guide_uavs=guide_uavs[:1],
        required_attack_uav_count=2,
        assigned_attack_uav_count=len(attack_uavs[:2]),
        start_position=Position(x=0.0, y=-16000.0),
        real_time_position=Position(x=0.0, y=-16000.0),
        target_center=cluster_0.center,
    )

    assignment_1 = ClusterAssignment(
        cluster_id=1,
        target_cluster=cluster_1,
        assigned_attack_uavs=attack_uavs[2:4],
        assigned_guide_uavs=guide_uavs[1:2],
        required_attack_uav_count=2,
        assigned_attack_uav_count=len(attack_uavs[2:4]),
        start_position=Position(x=0.0, y=-16000.0),
        real_time_position=Position(x=0.0, y=-16000.0),
        target_center=cluster_1.center,
    )

    allocation_plan = AllocationPlan(
        assignments=[assignment_0, assignment_1],
        resource_status=resource_status,
        algorithm_metadata=AlgorithmMetadata(
            algorithm_name="simple_resource_allocation",
            algorithm_type="rule_based",
            stage=PipelineStage.RESOURCE_ALLOCATION,
        ),
    )
    allocation_plan.validate()

    strike_order_0 = StrikeOrderPlan(
        cluster_id=0,
        ordered_target_ids=cluster_0.target_ids,
        path_positions=[target.position for target in cluster_0.targets],
        total_path_distance=None,
        expected_reward=None,
        algorithm_metadata=AlgorithmMetadata(
            algorithm_name="placeholder_dqn_strike_order",
            algorithm_type="dqn",
            stage=PipelineStage.STRIKE_ORDER_PLANNING,
        ),
    )
    strike_order_0.validate()

    strike_order_1 = StrikeOrderPlan(
        cluster_id=1,
        ordered_target_ids=cluster_1.target_ids,
        path_positions=[target.position for target in cluster_1.targets],
        total_path_distance=None,
        expected_reward=None,
        algorithm_metadata=AlgorithmMetadata(
            algorithm_name="placeholder_dqn_strike_order",
            algorithm_type="dqn",
            stage=PipelineStage.STRIKE_ORDER_PLANNING,
        ),
    )
    strike_order_1.validate()

    mission_plan = MissionPlan(
        screened_targets=screened_set,
        target_clusters=cluster_set,
        allocation_plan=allocation_plan,
        strike_order_plans={
            0: strike_order_0,
            1: strike_order_1,
        },
        algorithm_metadata=[
            screening_metadata,
            cluster_set.algorithm_metadata,
            allocation_plan.algorithm_metadata,
            strike_order_0.algorithm_metadata,
        ],
        metadata={"purpose": "contract_check"},
    )
    mission_plan.validate()

    snapshot = MissionTimeSnapshot(
        time=0.0,
        cluster_positions={
            0: assignment_0.real_time_position,
            1: assignment_1.real_time_position,
        },
        active_target_ids=[target.target_id for target in targets],
        destroyed_target_ids=[],
        active_uav_ids=[uav.uav_id for uav in uavs],
        damaged_uav_ids=[],
    )

    event = MissionEvent(
        event_time=300.0,
        event_type=MissionEventType.TARGET_DISAPPEARED,
        affected_target_ids=[selected_targets[0].target_id],
        payload={"reason": "debug_event"},
    )

    mission_metrics = MissionMetrics(
        total_targets=len(targets),
        selected_targets=len(selected_targets),
        destroyed_targets=0,
        total_uavs=len(uavs),
        lost_uavs=0,
        mission_completion_rate=0.0,
        num_dynamic_events=1,
        num_replanning=0,
    )

    mission_result = MissionResult(
        mission_plan=mission_plan,
        time_series=[snapshot],
        event_log=[event],
        metrics=mission_metrics,
        final_target_status={target.target_id: target.status.value for target in targets},
        final_uav_status={uav.uav_id: uav.status.value for uav in uavs},
        metadata={"source": "check_contracts.py"},
    )

    logger.info("Contracts built and validated successfully.")
    logger.info(f"Selected target ids: {[target.target_id for target in selected_targets]}")
    logger.info(f"Number of clusters: {mission_plan.target_clusters.num_clusters}")
    logger.info(f"Number of assignments: {mission_plan.allocation_plan.num_assignments}")
    logger.info(f"Mission event type: {mission_result.event_log[0].event_type}")
    logger.info(f"Mission metrics: {mission_result.metrics}")

    logger.info("Contracts check finished successfully.")


if __name__ == "__main__":
    main()