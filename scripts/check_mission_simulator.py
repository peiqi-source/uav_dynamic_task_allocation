"""check任务仿真器脚本，封装可直接运行的实验、检查或可视化流程。"""
from copy import deepcopy

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
from uav_dynamic_task_allocation.simulation.dynamic_events import (
    build_dynamic_event_manager,
)
from uav_dynamic_task_allocation.simulation.mission_simulator import (
    MissionSimulator,
    load_mission_simulator_config,
)
from uav_dynamic_task_allocation.simulation.mission_state import MissionRuntimeState
from uav_dynamic_task_allocation.utils.config import (
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


def build_demo_dynamic_config(config):
    """
    构造动态仿真 demo 配置。

    只修改内存中的配置，不写回 default.yaml。
    """
    local_config = deepcopy(config)

    local_config.setdefault("scenario", {})
    local_config["scenario"]["mode"] = "dynamic"
    local_config["scenario"]["time_step"] = 30
    local_config["scenario"]["max_time"] = 900

    local_config["scenario"].setdefault("dynamic_events", {})
    local_config["scenario"]["dynamic_events"]["enabled"] = True
    local_config["scenario"]["dynamic_events"]["event_schedule"] = [
        {
            "time": 150,
            "type": "target_disappeared",
            "affected_target_ids": [3],
            "description": "Demo target 3 disappeared before strike.",
        },
        {
            "time": 210,
            "type": "attack_uav_destroyed",
            "affected_uav_ids": [11],
            "description": "Demo attack UAV 11 destroyed during mission.",
        },
    ]

    local_config.setdefault("algorithm_policy", {})
    local_config["algorithm_policy"].setdefault("dynamic", {})
    local_config["algorithm_policy"]["dynamic"]["initial_planning"] = {
        "target_screening": "source_aligned",
        "destroy_target_selection": "weighted_kmeans",
        "target_clustering": "pso",
        "resource_allocation": "rule_based",
        "strike_order_planning": "dqn",
    }

    local_config.setdefault("mission_simulator", {})
    local_config["mission_simulator"]["default_uav_speed"] = 80.0
    local_config["mission_simulator"]["arrival_tolerance"] = 100.0
    local_config["mission_simulator"]["strike_targets_per_step"] = 1
    local_config["mission_simulator"]["stop_when_all_planned_targets_destroyed"] = True
    local_config["mission_simulator"].setdefault("output", {})
    local_config["mission_simulator"]["output"][
        "simulation_log_csv_path"
    ] = "outputs/simulation/mission_simulation_log.csv"
    local_config["mission_simulator"]["output"][
        "uav_trajectory_csv_path"
    ] = "outputs/simulation/uav_trajectory_log.csv"

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
    config = build_demo_dynamic_config(base_config)

    logger = setup_logger_from_config(config)

    logger.info("MissionSimulator check started.")

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

    initial_planner_result = planner.plan(
        battlefield_state=battlefield_state,
        scenario_mode="dynamic",
    )

    runtime_state = MissionRuntimeState.from_battlefield_state_and_plan(
        battlefield_state=battlefield_state,
        mission_plan=initial_planner_result.mission_plan,
        current_time=0.0,
    )

    event_manager = build_dynamic_event_manager(config)

    replanning_controller = ReplanningController(
        base_config=config,
        algorithm_policy=algorithm_policy,
        logger=logger,
    )

    simulator = MissionSimulator(
        config=load_mission_simulator_config(config),
        event_manager=event_manager,
        replanning_controller=replanning_controller,
        logger=logger,
    )

    result = simulator.run(
        runtime_state=runtime_state,
        original_battlefield_state=battlefield_state,
    )

    csv_path = simulator.write_log_csv(
        result=result,
        project_root=project_root,
    )

    uav_trajectory_path = simulator.write_uav_trajectory_csv(
        result=result,
        project_root=project_root,
    )

    logger.info("MissionSimulator finished successfully.")
    logger.info(f"Simulation log CSV: {csv_path}")
    logger.info(f"UAV trajectory CSV: {uav_trajectory_path}")
    logger.info(f"Simulation metadata: {result.metadata}")
    logger.info(f"Final state: {result.final_runtime_state.to_summary_dict()}")

    logger.info("MissionSimulator check finished successfully.")


if __name__ == "__main__":
    main()