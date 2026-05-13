from copy import deepcopy

from uav_dynamic_task_allocation.core.contracts import MissionEvent, MissionEventType
from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.planning.algorithm_policy import load_algorithm_policy
from uav_dynamic_task_allocation.planning.mission_planner import (
    MissionPlanner,
    load_mission_planner_config,
)
from uav_dynamic_task_allocation.simulation.mission_state import MissionRuntimeState
from uav_dynamic_task_allocation.utils.config import (
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


def build_demo_config(config):
    """
    构造 demo 配置，只修改内存，不写回 default.yaml。
    """
    local_config = deepcopy(config)

    local_config.setdefault("scenario", {})
    local_config["scenario"]["mode"] = "dynamic"

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


def main() -> None:
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    base_config = load_and_validate_config(config_path)
    config = build_demo_config(base_config)

    logger = setup_logger_from_config(config)

    logger.info("Runtime state check started.")

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

    planner_result = planner.plan(
        battlefield_state=battlefield_state,
        scenario_mode="dynamic",
    )

    runtime_state = MissionRuntimeState.from_battlefield_state_and_plan(
        battlefield_state=battlefield_state,
        mission_plan=planner_result.mission_plan,
        current_time=0.0,
    )

    logger.info("=" * 80)
    logger.info("Initial runtime state")
    logger.info(runtime_state.to_summary_dict())

    logger.info("=" * 80)
    logger.info("Sample assigned UAV runtime states")

    for uav_id in runtime_state.available_uav_ids[:10]:
        uav_state = runtime_state.get_uav_runtime_state(uav_id)

        logger.info(
            f"UAV {uav_id}: "
            f"status={uav_state.status}, "
            f"assigned_cluster_id={uav_state.assigned_cluster_id}, "
            f"position=({uav_state.current_position.x:.2f}, "
            f"{uav_state.current_position.y:.2f})"
        )

    logger.info("=" * 80)
    logger.info("Sample target runtime states")

    for target_id in runtime_state.available_target_ids[:10]:
        target_state = runtime_state.get_target_runtime_state(target_id)

        logger.info(
            f"Target {target_id}: "
            f"status={target_state.status}, "
            f"assigned_cluster_id={target_state.assigned_cluster_id}"
        )

    demo_events = [
        MissionEvent(
            event_time=150,
            event_type=MissionEventType.TARGET_DISAPPEARED,
            affected_target_ids=[3],
            affected_uav_ids=[],
            metadata={"description": "Demo target disappeared."},
        ),
        MissionEvent(
            event_time=210,
            event_type=MissionEventType.ATTACK_UAV_DESTROYED,
            affected_target_ids=[],
            affected_uav_ids=[11],
            metadata={"description": "Demo attack UAV destroyed."},
        ),
    ]

    for event in demo_events:
        logger.info("=" * 80)
        logger.info(
            f"Applying event: "
            f"time={event.event_time}, "
            f"type={event.event_type.value}, "
            f"targets={event.affected_target_ids}, "
            f"uavs={event.affected_uav_ids}"
        )

        runtime_state.apply_event(event)
        logger.info(runtime_state.to_summary_dict())

    logger.info("Runtime state check finished successfully.")


if __name__ == "__main__":
    main()