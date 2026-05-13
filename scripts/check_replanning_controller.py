"""check重规划controller脚本，封装可直接运行的实验、检查或可视化流程。"""
from copy import deepcopy

from uav_dynamic_task_allocation.core.contracts import MissionEvent, MissionEventType
from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.planning.algorithm_policy import load_algorithm_policy
from uav_dynamic_task_allocation.planning.mission_planner import (
    MissionPlanner,
    load_mission_planner_config,
)
from uav_dynamic_task_allocation.planning.replanning_controller import (
    ReplanningController,
)
from uav_dynamic_task_allocation.simulation.mission_state import MissionRuntimeState
from uav_dynamic_task_allocation.utils.config import (
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


def force_dynamic_demo_config(config):
    """
    构造用于检查 ReplanningController 的 demo 配置。

    这里只修改内存中的 config，不会写回 default.yaml。
    """
    local_config = deepcopy(config)

    local_config.setdefault("scenario", {})
    local_config["scenario"]["mode"] = "dynamic"

    # 为了降低 RF 模型缺失导致的检查失败风险，demo 里先用 weighted_kmeans。
    local_config.setdefault("algorithm_policy", {})
    local_config["algorithm_policy"].setdefault("dynamic", {})

    local_config["algorithm_policy"]["dynamic"]["initial_planning"] = {
        "target_screening": "source_aligned",
        "destroy_target_selection": "weighted_kmeans",
        "target_clustering": "pso",
        "resource_allocation": "rule_based",
        "strike_order_planning": "dqn",
    }

    return local_config


def build_strike_order_summary(mission_plan):
    """
    构造每个 cluster 的打击顺序摘要。

    单独写成函数，避免在 logger.info 里写复杂多行 f-string。
    """
    return {
        cluster_id: plan.ordered_target_ids
        for cluster_id, plan in mission_plan.strike_order_plans.items()
    }


def build_assignment_summary(mission_plan):
    """
    构造资源分配摘要。
    """
    summary = []

    for assignment in mission_plan.allocation_plan.assignments:
        summary.append(
            {
                "cluster_id": assignment.cluster_id,
                "targets": assignment.target_cluster.target_ids,
                "assigned_attack": assignment.assigned_attack_uav_ids,
                "assigned_guide": assignment.assigned_guide_uav_ids,
                "shortage": assignment.metadata.get("resource_shortage"),
            }
        )

    return summary


def main() -> None:
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        无返回值；通过状态变更、文件输出或日志记录体现执行结果。
    """
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    base_config = load_and_validate_config(config_path)
    config = force_dynamic_demo_config(base_config)

    logger = setup_logger_from_config(config)

    logger.info("ReplanningController check started.")

    data = load_all_data(config)

    battlefield_state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    algorithm_policy = load_algorithm_policy(config)

    planner = MissionPlanner(
        base_config=config,
        algorithm_policy=algorithm_policy,
        planner_config=load_mission_planner_config(config),
        logger=logger,
    )

    initial_result = planner.plan(
        battlefield_state=battlefield_state,
        scenario_mode="dynamic",
    )

    runtime_state = MissionRuntimeState.from_battlefield_state_and_plan(
        battlefield_state=battlefield_state,
        mission_plan=initial_result.mission_plan,
        current_time=0.0,
    )

    controller = ReplanningController(
        base_config=config,
        algorithm_policy=algorithm_policy,
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info("Initial mission plan summary")
    logger.info(f"Initial state: {runtime_state.to_summary_dict()}")
    logger.info(f"Initial plan metadata: {initial_result.mission_plan.metadata}")
    logger.info(
        f"Initial strike orders: "
        f"{build_strike_order_summary(initial_result.mission_plan)}"
    )

    demo_events = [
        MissionEvent(
            event_time=300,
            event_type=MissionEventType.TARGET_DISAPPEARED,
            affected_target_ids=[3],
            affected_uav_ids=[],
            metadata={"description": "Demo target disappeared."},
        ),
        MissionEvent(
            event_time=600,
            event_type=MissionEventType.ATTACK_UAV_DESTROYED,
            affected_target_ids=[],
            affected_uav_ids=[11],
            metadata={"description": "Demo attack UAV destroyed."},
        ),
    ]

    for event in demo_events:
        logger.info("=" * 80)
        logger.info(
            "Applying event before replanning: "
            f"time={event.event_time}, "
            f"type={event.event_type.value}, "
            f"affected_targets={event.affected_target_ids}, "
            f"affected_uavs={event.affected_uav_ids}"
        )

        runtime_state.apply_event(event)

        result = controller.handle_event(
            runtime_state=runtime_state,
            event=event,
            original_battlefield_state=battlefield_state,
        )

        logger.info(f"Replanning result metadata: {result.metadata}")
        logger.info(f"Runtime state: {runtime_state.to_summary_dict()}")
        logger.info(f"Updated plan metadata: {result.updated_mission_plan.metadata}")
        logger.info(
            f"Updated assignments: "
            f"{build_assignment_summary(result.updated_mission_plan)}"
        )
        logger.info(
            f"Updated strike orders: "
            f"{build_strike_order_summary(result.updated_mission_plan)}"
        )

    logger.info("ReplanningController check finished successfully.")


if __name__ == "__main__":
    main()