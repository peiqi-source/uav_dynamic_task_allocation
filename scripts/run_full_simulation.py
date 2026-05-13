"""runfull仿真脚本，封装可直接运行的实验、检查或可视化流程。"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.core.contracts import MissionEvent, MissionEventType
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.planning.algorithm_policy import load_algorithm_policy
from uav_dynamic_task_allocation.planning.mission_planner import (
    MissionPlanner,
    load_mission_planner_config,
)
from uav_dynamic_task_allocation.planning.replanning_controller import (
    ReplanningController,
)
from uav_dynamic_task_allocation.planning.support_policy import (
    SupportDecision,
    SupportPolicy,
    load_support_policy_config,
)
from uav_dynamic_task_allocation.simulation.dynamic_events import (
    build_dynamic_event_manager,
)
from uav_dynamic_task_allocation.simulation.mission_metrics import (
    build_mission_metrics_analyzer,
)
from uav_dynamic_task_allocation.simulation.mission_simulator import (
    MissionSimulator,
    load_mission_simulator_config,
)
from uav_dynamic_task_allocation.simulation.mission_state import MissionRuntimeState
from uav_dynamic_task_allocation.simulation.mission_visualization import (
    build_mission_visualizer,
)
from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run full UAV dynamic task allocation simulation."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file.",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        choices=["static", "dynamic"],
        default=None,
        help="Override scenario.mode.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override full_simulation.run_name.",
    )

    parser.add_argument(
        "--demo-events",
        action="store_true",
        help="Force using demo dynamic events defined in full_simulation.demo_dynamic_events.",
    )

    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable metrics analysis.",
    )

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable visualization.",
    )

    return parser.parse_args()


def set_nested_config_value(
    config: dict[str, Any],
    key_path: str,
    value: Any,
) -> None:
    """修改嵌套配置。"""
    keys = key_path.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def build_run_directory(
    project_root: Path,
    config: dict[str, Any],
    run_name_override: str | None = None,
) -> Path:
    """
    根据配置创建本次实验输出目录。

    示例：
        outputs/full_simulation/debug_full_simulation_20260507_210000
    """
    output_root = Path(
        get_config_value(
            config,
            "full_simulation.output_root",
            default="outputs/full_simulation",
        )
    )

    if not output_root.is_absolute():
        output_root = project_root / output_root

    run_name = run_name_override or str(
        get_config_value(
            config,
            "full_simulation.run_name",
            default="debug_full_simulation",
        )
    )

    create_timestamped = bool(
        get_config_value(
            config,
            "full_simulation.create_timestamped_run_dir",
            default=True,
        )
    )

    if create_timestamped:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / f"{run_name}_{timestamp}"
    else:
        run_dir = output_root / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def configure_output_paths(
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """
    把本次实验的输出路径统一改到 run_dir 下。

    这样每次 full simulation 都不会覆盖旧结果。
    """
    local_config = deepcopy(config)

    simulation_dir = run_dir / "simulation"
    figure_dir = simulation_dir / "figures"

    simulation_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    simulation_log_path = simulation_dir / "mission_simulation_log.csv"
    uav_trajectory_path = simulation_dir / "uav_trajectory_log.csv"
    summary_path = simulation_dir / "mission_simulation_summary.csv"
    timeline_path = simulation_dir / "mission_simulation_timeline_metrics.csv"
    resource_assessment_path = simulation_dir / "resource_assessment.csv"
    support_decision_path = simulation_dir / "support_decisions.csv"

    set_nested_config_value(
        local_config,
        "mission_simulator.output.simulation_log_csv_path",
        str(simulation_log_path),
    )
    set_nested_config_value(
        local_config,
        "mission_simulator.output.uav_trajectory_csv_path",
        str(uav_trajectory_path),
    )

    set_nested_config_value(
        local_config,
        "mission_metrics.input_csv_path",
        str(simulation_log_path),
    )
    set_nested_config_value(
        local_config,
        "mission_metrics.output.summary_csv_path",
        str(summary_path),
    )
    set_nested_config_value(
        local_config,
        "mission_metrics.output.timeline_csv_path",
        str(timeline_path),
    )
    set_nested_config_value(
        local_config,
        "mission_metrics.output.figure_dir",
        str(figure_dir),
    )

    set_nested_config_value(
        local_config,
        "mission_visualization.input.uav_trajectory_csv_path",
        str(uav_trajectory_path),
    )
    set_nested_config_value(
        local_config,
        "mission_visualization.input.simulation_log_csv_path",
        str(simulation_log_path),
    )
    set_nested_config_value(
        local_config,
        "mission_visualization.output.figure_dir",
        str(figure_dir),
    )
    set_nested_config_value(
        local_config,
        "monte_carlo.output_csv_path",
        str(resource_assessment_path),
    )
    set_nested_config_value(
        local_config,
        "monte_carlo.figure_path",
        str(figure_dir / "monte_carlo_completion_distribution.png"),
    )
    set_nested_config_value(
        local_config,
        "support_policy.output_csv_path",
        str(support_decision_path),
    )

    return local_config


def apply_runtime_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    应用命令行参数和 demo event 覆盖项。
    """
    local_config = deepcopy(config)

    if args.scenario is not None:
        set_nested_config_value(local_config, "scenario.mode", args.scenario)

    if args.demo_events:
        demo_events = list(
            get_config_value(
                local_config,
                "full_simulation.demo_dynamic_events",
                default=[
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
                ],
            )
        )

        set_nested_config_value(local_config, "scenario.mode", "dynamic")
        set_nested_config_value(local_config, "scenario.dynamic_events.enabled", True)
        set_nested_config_value(
            local_config,
            "scenario.dynamic_events.event_schedule",
            demo_events,
        )

    if args.no_metrics:
        set_nested_config_value(local_config, "full_simulation.run_metrics", False)

    if args.no_visualization:
        set_nested_config_value(
            local_config,
            "full_simulation.run_visualization",
            False,
        )

    return local_config


def save_effective_config(
    config: dict[str, Any],
    run_dir: Path,
    original_config_path: Path,
) -> None:
    """
    保存本次运行使用的最终配置，方便实验复现。
    """
    effective_config_path = run_dir / "effective_config.yaml"

    with effective_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    copied_config_path = run_dir / "source_config.yaml"

    if original_config_path.exists():
        shutil.copy2(original_config_path, copied_config_path)


def build_final_summary(
    planner_result,
    simulation_result,
    metrics_result,
    figure_paths,
    run_dir: Path,
    initial_support_decisions: list[SupportDecision] | None = None,
) -> dict[str, Any]:
    """
    构造最终 summary，便于日志输出和保存。
    """
    summary = {
        "run_dir": str(run_dir),
        "planner_metadata": planner_result.metadata,
        "simulation_metadata": simulation_result.metadata,
        "final_runtime_state": simulation_result.final_runtime_state.to_summary_dict(),
        "num_replanning_results": len(simulation_result.replanning_results),
        "figure_paths": [str(path) for path in figure_paths],
        "num_initial_support_decisions": len(initial_support_decisions or []),
        "initial_support_decisions": [
            decision.to_dict() for decision in (initial_support_decisions or [])
        ],
    }

    if metrics_result is not None:
        summary["metrics_summary"] = metrics_result.summary

    return summary


def save_final_summary(
    summary: dict[str, Any],
    run_dir: Path,
) -> Path:
    """保存最终 summary JSON。"""
    summary_path = run_dir / "final_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return summary_path


def write_dynamic_events_csv(
    simulation_result,
    run_dir: Path,
    initial_events: list[MissionEvent] | None = None,
) -> Path:
    """Save dynamic event records from replanning results."""
    path = run_dir / "simulation" / "dynamic_events.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "event_time",
        "event_type",
        "affected_target_ids",
        "affected_uav_ids",
        "metadata",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for event in initial_events or []:
            writer.writerow(
                {
                    "event_time": event.event_time,
                    "event_type": event.event_type.value,
                    "affected_target_ids": event.affected_target_ids,
                    "affected_uav_ids": event.affected_uav_ids,
                    "metadata": event.metadata,
                }
            )
        for replanning_result in simulation_result.replanning_results:
            event = replanning_result.event
            writer.writerow(
                {
                    "event_time": event.event_time,
                    "event_type": event.event_type.value,
                    "affected_target_ids": event.affected_target_ids,
                    "affected_uav_ids": event.affected_uav_ids,
                    "metadata": event.metadata,
                }
            )
    return path


def write_support_decisions_csv(
    simulation_result,
    run_dir: Path,
    initial_support_decisions: list[SupportDecision] | None = None,
) -> Path:
    """Save support decisions from replanning results."""
    path = run_dir / "simulation" / "support_decisions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "support_required",
        "support_type",
        "event_id",
        "donor_cluster_id",
        "receiver_cluster_id",
        "reassigned_uav_ids",
        "affected_target_ids",
        "reason",
        "estimated_completion_rate_before",
        "estimated_completion_rate_after",
        "updated_plan",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for decision in initial_support_decisions or []:
            writer.writerow(decision.to_dict())
        for replanning_result in simulation_result.replanning_results:
            decision = getattr(replanning_result, "support_decision", None)
            if decision is not None:
                writer.writerow(decision.to_dict())
    return path


def trigger_initial_resource_support(
    config: dict[str, Any],
    planner_result,
    logger,
) -> tuple[list[MissionEvent], list[SupportDecision]]:
    """Trigger support at t=0 when Monte Carlo assessment is below threshold."""
    assessment = planner_result.mission_plan.allocation_plan.metadata.get(
        "resource_assessment",
        [],
    )
    risky_items = [
        item for item in assessment
        if bool(item.get("support_required")) or not bool(item.get("threshold_passed", True))
    ]
    if not risky_items:
        logger.info("Monte Carlo assessment passed for all clusters; no initial support event.")
        return [], []

    support_config = load_support_policy_config(config)
    if not support_config.enabled:
        logger.info(
            "Monte Carlo assessment found risky clusters, but support policy is disabled."
        )
        return [], []

    affected_target_ids: list[int] = []
    for item in risky_items:
        affected_target_ids.extend(
            int(target_id)
            for target_id in item.get("details", {}).get("target_ids", [])
        )
    affected_target_ids = sorted(set(affected_target_ids))

    event = MissionEvent(
        event_time=0.0,
        event_type=MissionEventType.RESOURCE_SHORTAGE,
        affected_target_ids=affected_target_ids,
        metadata={
            "source": "initial_monte_carlo_resource_assessment",
            "risky_cluster_ids": [item.get("cluster_id") for item in risky_items],
            "completion_threshold": get_config_value(
                config,
                "monte_carlo.completion_threshold",
                default=0.8,
            ),
        },
    )
    policy = SupportPolicy(support_config)
    decision = policy.decide_and_apply(
        event=event,
        allocation_plan=planner_result.mission_plan.allocation_plan,
    )
    planner_result.mission_plan.metadata.setdefault("support_decisions", []).append(
        decision.to_dict()
    )
    planner_result.metadata.setdefault("initial_support_decisions", []).append(
        decision.to_dict()
    )
    logger.info(
        "Monte Carlo assessment below threshold; initial support policy triggered: "
        f"{decision.to_dict()}"
    )
    return [event], [decision]


def main() -> None:
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        无返回值；通过状态变更、文件输出或日志记录体现执行结果。
    """
    args = parse_args()

    project_root = get_project_root()
    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = project_root / config_path

    base_config = load_and_validate_config(config_path)
    base_config = apply_runtime_overrides(base_config, args)

    run_dir = build_run_directory(
        project_root=project_root,
        config=base_config,
        run_name_override=args.run_name,
    )

    config = configure_output_paths(
        config=base_config,
        run_dir=run_dir,
    )

    save_effective_config(
        config=config,
        run_dir=run_dir,
        original_config_path=config_path,
    )

    logger = setup_logger_from_config(config)

    logger.info("=" * 100)
    logger.info("Full mission simulation started.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Scenario mode: {get_config_value(config, 'scenario.mode', default='static')}")
    logger.info(
        "Dynamic events enabled: "
        f"{get_config_value(config, 'scenario.dynamic_events.enabled', default=False)}"
    )
    logger.info("=" * 100)

    # 1. 加载数据并构造 BattlefieldState
    data = load_all_data(config)

    battlefield_state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    logger.info(
        "BattlefieldState built: "
        f"num_uavs={len(battlefield_state.uavs)}, "
        f"num_targets={len(battlefield_state.targets)}"
    )

    # 2. 算法策略和任务规划
    algorithm_policy = load_algorithm_policy(config)

    mission_planner = MissionPlanner(
        base_config=config,
        algorithm_policy=algorithm_policy,
        planner_config=load_mission_planner_config(config),
        logger=logger,
    )

    planner_result = mission_planner.plan(
        battlefield_state=battlefield_state,
        scenario_mode=get_config_value(config, "scenario.mode", default=None),
    )

    planner_summary_csv = mission_planner.write_debug_csv(
        result=planner_result,
        output_path=run_dir / "mission_planner_summary.csv",
        project_root=project_root,
    )

    logger.info(f"Mission planner summary CSV: {planner_summary_csv}")

    initial_support_events, initial_support_decisions = trigger_initial_resource_support(
        config=config,
        planner_result=planner_result,
        logger=logger,
    )

    # 3. 初始化运行时状态
    runtime_state = MissionRuntimeState.from_battlefield_state_and_plan(
        battlefield_state=battlefield_state,
        mission_plan=planner_result.mission_plan,
        current_time=0.0,
    )

    logger.info(f"Initial runtime state: {runtime_state.to_summary_dict()}")

    # 4. 动态事件管理器和重规划控制器
    event_manager = build_dynamic_event_manager(config)

    replanning_controller = ReplanningController(
        base_config=config,
        algorithm_policy=algorithm_policy,
        logger=logger,
    )

    # 5. 执行仿真
    simulator = MissionSimulator(
        config=load_mission_simulator_config(config),
        event_manager=event_manager,
        replanning_controller=replanning_controller,
        logger=logger,
    )

    simulation_result = simulator.run(
        runtime_state=runtime_state,
        original_battlefield_state=battlefield_state,
    )

    simulation_log_path = simulator.write_log_csv(
        result=simulation_result,
        project_root=project_root,
    )

    uav_trajectory_path = simulator.write_uav_trajectory_csv(
        result=simulation_result,
        project_root=project_root,
    )

    logger.info(f"Simulation log CSV: {simulation_log_path}")
    logger.info(f"UAV trajectory CSV: {uav_trajectory_path}")

    dynamic_events_csv = write_dynamic_events_csv(
        simulation_result=simulation_result,
        run_dir=run_dir,
        initial_events=initial_support_events,
    )
    support_decisions_csv = write_support_decisions_csv(
        simulation_result=simulation_result,
        run_dir=run_dir,
        initial_support_decisions=initial_support_decisions,
    )
    logger.info(f"Dynamic events CSV: {dynamic_events_csv}")
    logger.info(f"Support decisions CSV: {support_decisions_csv}")

    # 6. 指标分析
    metrics_result = None
    metric_figure_paths = []

    run_metrics = bool(
        get_config_value(
            config,
            "full_simulation.run_metrics",
            default=True,
        )
    )

    if run_metrics:
        metrics_analyzer = build_mission_metrics_analyzer(config)
        metrics_result = metrics_analyzer.analyze(project_root=project_root)

        summary_path, timeline_path = metrics_analyzer.write_outputs(
            result=metrics_result,
            project_root=project_root,
        )

        metric_figure_paths = metrics_analyzer.save_plots(
            result=metrics_result,
            project_root=project_root,
        )

        logger.info(f"Metrics summary CSV: {summary_path}")
        logger.info(f"Metrics timeline CSV: {timeline_path}")
        logger.info(f"Metrics figure paths: {metric_figure_paths}")
        logger.info(f"Metrics summary: {metrics_result.summary}")

    # 7. 可视化
    figure_paths = []

    run_visualization = bool(
        get_config_value(
            config,
            "full_simulation.run_visualization",
            default=True,
        )
    )

    if run_visualization:
        visualizer = build_mission_visualizer(config)

        figure_paths = visualizer.save_all(
            targets=battlefield_state.targets,
            project_root=project_root,
        )

        for figure_path in figure_paths:
            logger.info(f"Visualization figure saved: {figure_path}")

    # 8. 保存最终 summary
    all_figure_paths = [*metric_figure_paths, *figure_paths]
    final_summary = build_final_summary(
        planner_result=planner_result,
        simulation_result=simulation_result,
        metrics_result=metrics_result,
        figure_paths=all_figure_paths,
        run_dir=run_dir,
        initial_support_decisions=initial_support_decisions,
    )

    final_summary_path = save_final_summary(
        summary=final_summary,
        run_dir=run_dir,
    )

    logger.info(f"Final summary JSON: {final_summary_path}")
    logger.info(f"Final summary: {final_summary}")
    logger.info("Full mission simulation finished successfully.")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
