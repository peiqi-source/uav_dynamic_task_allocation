"""check动态事件集合脚本，封装可直接运行的实验、检查或可视化流程。"""
from copy import deepcopy

from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.planning.algorithm_policy import (
    load_algorithm_policy,
)
from uav_dynamic_task_allocation.planning.mission_planner import (
    MissionPlanner,
    load_mission_planner_config,
)
from uav_dynamic_task_allocation.simulation.dynamic_events import (
    build_dynamic_event_manager,
)
from uav_dynamic_task_allocation.simulation.mission_state import MissionRuntimeState
from uav_dynamic_task_allocation.utils.config import (
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


def ensure_demo_dynamic_events(config):
    """
    如果 default.yaml 中没有启用动态事件，这里临时注入一组 demo 事件。

    这个函数只用于检查脚本，不会修改你的 YAML 文件。
    """
    local_config = deepcopy(config)

    if "scenario" not in local_config:
        local_config["scenario"] = {}

    local_config["scenario"]["mode"] = "dynamic"

    if "dynamic_events" not in local_config["scenario"]:
        local_config["scenario"]["dynamic_events"] = {}

    local_config["scenario"]["dynamic_events"]["enabled"] = True
    local_config["scenario"]["dynamic_events"]["event_schedule"] = [
        {
            "time": 300,
            "type": "target_disappeared",
            "affected_target_ids": [3],
            "description": "Demo target disappeared event.",
        },
        {
            "time": 600,
            "type": "attack_uav_destroyed",
            "affected_uav_ids": [11],
            "description": "Demo attack UAV destroyed event.",
        },
        {
            "time": 900,
            "type": "guide_uav_destroyed",
            "affected_uav_ids": [5],
            "description": "Demo guide UAV destroyed event.",
        },
    ]

    return local_config


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
    config = ensure_demo_dynamic_events(base_config)

    logger = setup_logger_from_config(config)

    logger.info("Dynamic events check started.")

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
    )

    mission_state = MissionRuntimeState.from_battlefield_state_and_plan(
        battlefield_state=battlefield_state,
        mission_plan=planner_result.mission_plan,
        current_time=0.0,
    )

    event_manager = build_dynamic_event_manager(config)

    logger.info(f"Initial mission state: {mission_state.to_summary_dict()}")
    logger.info(
        "Loaded dynamic events: "
        f"{[(event.event_time, event.event_type.value) for event in event_manager.events]}"
    )

    time_step = 300
    max_time = 900

    previous_time = 0.0

    for current_time in range(time_step, max_time + time_step, time_step):
        events = event_manager.get_events_between(
            previous_time=previous_time,
            current_time=float(current_time),
        )

        logger.info("=" * 80)
        logger.info(f"Time advanced: {previous_time} -> {current_time}")
        logger.info(f"Triggered events: {[event.event_type.value for event in events]}")

        for event in events:
            logger.info(
                f"Applying event: "
                f"time={event.event_time}, "
                f"type={event.event_type.value}, "
                f"targets={event.affected_target_ids}, "
                f"uavs={event.affected_uav_ids}, "
                f"metadata={event.metadata}"
            )

            mission_state.apply_event(event)

            decisions = algorithm_policy.get_replanning_decisions(
                event_type=event.event_type,
                scenario_mode="dynamic",
            )

            logger.info(
                "Replanning policy decisions: "
                f"{algorithm_policy.summarize_decisions(decisions)}"
            )

        logger.info(f"Mission state summary: {mission_state.to_summary_dict()}")
        previous_time = float(current_time)

    logger.info("Dynamic events check finished successfully.")


if __name__ == "__main__":
    main()