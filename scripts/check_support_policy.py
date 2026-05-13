from copy import deepcopy

from uav_dynamic_task_allocation.core.contracts import MissionEvent, MissionEventType
from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.planning.algorithm_policy import load_algorithm_policy
from uav_dynamic_task_allocation.planning.mission_planner import (
    MissionPlanner,
    load_mission_planner_config,
)
from uav_dynamic_task_allocation.planning.support_policy import (
    SupportPolicy,
    load_support_policy_config,
)
from uav_dynamic_task_allocation.utils.config import (
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


def force_dynamic_demo_config(config):
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
    base_config = load_and_validate_config(project_root / "configs" / "default.yaml")
    config = force_dynamic_demo_config(base_config)
    logger = setup_logger_from_config(config)

    logger.info("Support policy check started.")

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
    mission_plan = planner.plan(
        battlefield_state=battlefield_state,
        scenario_mode="dynamic",
    ).mission_plan

    event = MissionEvent(
        event_time=600,
        event_type=MissionEventType.ATTACK_UAV_DESTROYED,
        affected_uav_ids=[mission_plan.allocation_plan.assignments[0].assigned_attack_uav_ids[0]],
        metadata={"description": "Support policy check event."},
    )

    policy = SupportPolicy(load_support_policy_config(config))
    decision = policy.decide_and_apply(
        event=event,
        allocation_plan=mission_plan.allocation_plan,
    )
    csv_path = policy.write_csv([decision], project_root=project_root)

    logger.info(f"Support decision CSV: {csv_path}")
    logger.info(f"Support decision: {decision.to_dict()}")
    logger.info("Support policy check finished successfully.")


if __name__ == "__main__":
    main()
