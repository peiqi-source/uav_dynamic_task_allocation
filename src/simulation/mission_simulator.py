from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from uav_dynamic_task_allocation.core.contracts import MissionEvent, MissionPlan
from uav_dynamic_task_allocation.core.entities import Position
from uav_dynamic_task_allocation.planning.replanning_controller import (
    ReplanningController,
)
from uav_dynamic_task_allocation.simulation.dynamic_events import DynamicEventManager
from uav_dynamic_task_allocation.simulation.mission_state import MissionRuntimeState
from uav_dynamic_task_allocation.utils.config import get_config_value, resolve_path


class MissionSimulatorError(Exception):
    """任务仿真过程中的自定义错误。"""


@dataclass(frozen=True)
class MissionSimulatorConfig:
    """
    任务仿真配置。

    time_step:
        每次仿真推进的时间长度。

    max_time:
        最大仿真时间。

    default_uav_speed:
        UAV 小组默认飞行速度。当前先使用统一速度，后续可以扩展为每架 UAV 自身速度。

    arrival_tolerance:
        当 UAV 小组距离目标群中心小于该阈值时，认为已经到达目标群任务区。

    strike_targets_per_step:
        UAV 小组到达目标群后，每个时间步最多打击多少个目标。
    """

    time_step: float = 30.0
    max_time: float = 2200.0

    default_uav_speed: float = 80.0
    arrival_tolerance: float = 100.0
    strike_targets_per_step: int = 1

    stop_when_all_planned_targets_destroyed: bool = True

    simulation_log_csv_path: str = "outputs/simulation/mission_simulation_log.csv"
    uav_trajectory_csv_path: str = "outputs/simulation/uav_trajectory_log.csv"

    def validate(self) -> None:
        if self.time_step <= 0:
            raise MissionSimulatorError("time_step must be positive.")

        if self.max_time <= 0:
            raise MissionSimulatorError("max_time must be positive.")

        if self.default_uav_speed <= 0:
            raise MissionSimulatorError("default_uav_speed must be positive.")

        if self.arrival_tolerance < 0:
            raise MissionSimulatorError("arrival_tolerance must be non-negative.")

        if self.strike_targets_per_step <= 0:
            raise MissionSimulatorError("strike_targets_per_step must be positive.")


@dataclass
class ClusterExecutionState:
    """
    单个目标群的执行状态。

    一个 cluster 对应一个 UAV 小组。
    UAV 小组先从 start_position 飞向 target_center；
    到达后按 StrikeOrderPlan 中的 ordered_target_ids 逐个打击目标。
    """

    cluster_id: int
    current_position: Position
    target_center: Position
    assigned_attack_uav_ids: list[int]
    assigned_guide_uav_ids: list[int]
    ordered_target_ids: list[int]

    status: str = "en_route"
    current_order_index: int = 0
    reached_time: float | None = None
    completed_time: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_target_ids(self) -> list[int]:
        return self.ordered_target_ids[self.current_order_index :]

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"


@dataclass
class SimulationStepRecord:
    """单个仿真时间步记录。"""

    time: float
    event_types: list[str]
    replanning_count: int
    destroyed_target_ids: list[int]
    disappeared_target_ids: list[int]
    damaged_uav_ids: list[int]
    available_target_count: int
    available_uav_count: int
    completed_cluster_count: int
    total_cluster_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class UAVTrajectoryRecord:
    """单架 UAV 在某个时间步的轨迹状态记录。"""

    time: float
    uav_id: int
    status: str
    assigned_cluster_id: int | None
    x: float
    y: float
    is_damaged: bool


@dataclass
class MissionSimulationResult:
    """任务仿真结果。"""

    final_runtime_state: MissionRuntimeState
    final_mission_plan: MissionPlan
    step_records: list[SimulationStepRecord]
    uav_trajectory_records: list[UAVTrajectoryRecord]
    replanning_results: list[Any]
    metadata: dict[str, Any]


class MissionSimulator:
    """
    任务仿真器。

    它负责真正的任务时间推进：

    1. UAV 小组按速度向目标群中心移动；
    2. 到达目标群后，根据 StrikeOrderPlan 逐个摧毁目标；
    3. 动态事件按时间触发；
    4. 事件触发后调用 ReplanningController；
    5. 记录每个时间步的任务状态。
    """

    def __init__(
        self,
        config: MissionSimulatorConfig,
        event_manager: DynamicEventManager,
        replanning_controller: ReplanningController,
        logger: logging.Logger | None = None,
    ) -> None:
        config.validate()

        self.config = config
        self.event_manager = event_manager
        self.replanning_controller = replanning_controller
        self.logger = logger or logging.getLogger(__name__)

        self.cluster_states: dict[int, ClusterExecutionState] = {}
        self.replanning_results: list[Any] = []

    def run(
        self,
        runtime_state: MissionRuntimeState,
        original_battlefield_state=None,
    ) -> MissionSimulationResult:
        """
        执行任务仿真。

        runtime_state 中应包含初始 MissionPlan。
        """
        runtime_state.validate()

        runtime_state.sync_assignments_from_mission_plan(runtime_state.mission_plan)

        self._sync_cluster_states_from_plan(
            mission_plan=runtime_state.mission_plan,
            runtime_state=runtime_state,
        )

        step_records: list[SimulationStepRecord] = []
        uav_trajectory_records: list[UAVTrajectoryRecord] = []

        previous_time = float(runtime_state.current_time)
        current_time = previous_time

        self.logger.info("Mission simulation started.")
        self.logger.info(f"Initial state: {runtime_state.to_summary_dict()}")

        while current_time < self.config.max_time:
            next_time = min(
                current_time + self.config.time_step,
                self.config.max_time,
            )

            # 1. UAV 小组运动和打击执行
            destroyed_this_step = self._advance_cluster_execution(
                runtime_state=runtime_state,
                delta_time=next_time - current_time,
                current_time=next_time,
            )

            # 2. 动态事件触发
            events = self.event_manager.get_events_between(
                previous_time=current_time,
                current_time=next_time,
            )

            # 3. 应用事件并重规划
            for event in events:
                self._handle_event(
                    runtime_state=runtime_state,
                    event=event,
                    original_battlefield_state=original_battlefield_state,
                )

            runtime_state.current_time = next_time

            # 4. 记录当前时间步状态
            step_record = self._build_step_record(
                runtime_state=runtime_state,
                current_time=next_time,
                events=events,
                destroyed_this_step=destroyed_this_step,
            )
            step_records.append(step_record)
            uav_trajectory_records.extend(
                self._build_uav_trajectory_records(
                    runtime_state=runtime_state,
                    current_time=next_time,
                )
            )

            self.logger.info(
                "Simulation step: "
                f"time={next_time}, "
                f"events={[event.event_type.value for event in events]}, "
                f"destroyed_this_step={destroyed_this_step}, "
                f"state={runtime_state.to_summary_dict()}"
            )

            previous_time = current_time
            current_time = next_time

            if self._should_stop(runtime_state):
                self.logger.info(
                    "Mission simulation stopped because all planned targets "
                    "have been destroyed."
                )
                break

        metadata = {
            "final_time": runtime_state.current_time,
            "num_steps": len(step_records),
            "num_replanning": len(self.replanning_results),
            "destroyed_target_ids": sorted(runtime_state.destroyed_target_ids),
            "disappeared_target_ids": sorted(runtime_state.disappeared_target_ids),
            "damaged_uav_ids": sorted(runtime_state.damaged_uav_ids),
            "completed_cluster_ids": [
                cluster_id
                for cluster_id, state in self.cluster_states.items()
                if state.is_completed
            ],
        }

        self.logger.info(f"Mission simulation finished: {metadata}")

        return MissionSimulationResult(
            final_runtime_state=runtime_state,
            final_mission_plan=runtime_state.mission_plan,
            step_records=step_records,
            uav_trajectory_records=uav_trajectory_records,
            replanning_results=self.replanning_results,
            metadata=metadata,
        )

    def write_log_csv(
        self,
        result: MissionSimulationResult,
        output_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> Path:
        """保存仿真时间步日志。"""
        path = resolve_path(
            output_path or self.config.simulation_log_csv_path,
            project_root=project_root,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "time",
            "event_types",
            "replanning_count",
            "destroyed_target_ids",
            "disappeared_target_ids",
            "damaged_uav_ids",
            "available_target_count",
            "available_uav_count",
            "completed_cluster_count",
            "total_cluster_count",
            "metadata",
        ]

        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for record in result.step_records:
                writer.writerow(
                    {
                        "time": record.time,
                        "event_types": record.event_types,
                        "replanning_count": record.replanning_count,
                        "destroyed_target_ids": record.destroyed_target_ids,
                        "disappeared_target_ids": record.disappeared_target_ids,
                        "damaged_uav_ids": record.damaged_uav_ids,
                        "available_target_count": record.available_target_count,
                        "available_uav_count": record.available_uav_count,
                        "completed_cluster_count": record.completed_cluster_count,
                        "total_cluster_count": record.total_cluster_count,
                        "metadata": record.metadata,
                    }
                )

        return path

    def write_uav_trajectory_csv(
            self,
            result: MissionSimulationResult,
            output_path: str | Path | None = None,
            project_root: str | Path | None = None,
    ) -> Path:
        """保存单架 UAV 轨迹日志。"""
        path = resolve_path(
            output_path or self.config.uav_trajectory_csv_path,
            project_root=project_root,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "time",
            "uav_id",
            "status",
            "assigned_cluster_id",
            "x",
            "y",
            "is_damaged",
        ]

        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for record in result.uav_trajectory_records:
                writer.writerow(
                    {
                        "time": record.time,
                        "uav_id": record.uav_id,
                        "status": record.status,
                        "assigned_cluster_id": record.assigned_cluster_id,
                        "x": record.x,
                        "y": record.y,
                        "is_damaged": record.is_damaged,
                    }
                )

        return path

    def _advance_cluster_execution(
            self,
            runtime_state: MissionRuntimeState,
            delta_time: float,
            current_time: float,
    ) -> list[int]:
        """
        推进所有目标群的执行状态。

        当前版本已经接入 UAVRuntimeState：
        - 每架 UAV 按自己的 current_position 和 speed 向目标群中心移动；
        - cluster_state.current_position 只是 UAV 小组当前位置的质心；
        - 到达判定基于该 cluster 内攻击 UAV 的实时位置；
        - UAV 损毁后不再参与移动和打击。
        """
        destroyed_this_step: list[int] = []

        for cluster_state in self.cluster_states.values():
            if cluster_state.is_completed:
                continue

            active_attack_uavs = [
                runtime_state.get_uav_runtime_state(uav_id)
                for uav_id in cluster_state.assigned_attack_uav_ids
                if uav_id in runtime_state.uav_runtime_states
                   and not runtime_state.get_uav_runtime_state(uav_id).is_damaged
            ]

            if not active_attack_uavs:
                cluster_state.status = "resource_shortage"
                cluster_state.metadata.setdefault("resource_shortage_times", []).append(
                    current_time
                )
                continue

            if cluster_state.status == "en_route":
                self._move_cluster_uavs_towards_target(
                    runtime_state=runtime_state,
                    cluster_state=cluster_state,
                    delta_time=delta_time,
                    current_time=current_time,
                )

                self._update_cluster_position_from_uavs(
                    runtime_state=runtime_state,
                    cluster_state=cluster_state,
                )

                if self._cluster_has_arrived(
                        runtime_state=runtime_state,
                        cluster_state=cluster_state,
                ):
                    cluster_state.status = "striking"
                    cluster_state.reached_time = current_time

                    for uav_state in active_attack_uavs:
                        uav_state.status = "striking"

            if cluster_state.status == "striking":
                newly_destroyed = self._execute_strike_step(
                    runtime_state=runtime_state,
                    cluster_state=cluster_state,
                    current_time=current_time,
                )
                destroyed_this_step.extend(newly_destroyed)

        return destroyed_this_step

    def _move_cluster_uavs_towards_target(
            self,
            runtime_state: MissionRuntimeState,
            cluster_state: ClusterExecutionState,
            delta_time: float,
            current_time: float,
    ) -> None:
        """
        按 UAVRuntimeState 推进每架 UAV 的位置。

        这里不再直接移动 cluster_state.current_position，
        而是移动分配给该 cluster 的每一架 UAV。
        """
        assigned_uav_ids = self._get_cluster_assigned_uav_ids(cluster_state)

        for uav_id in assigned_uav_ids:
            if uav_id not in runtime_state.uav_runtime_states:
                continue

            uav_state = runtime_state.get_uav_runtime_state(uav_id)

            if uav_state.is_damaged:
                continue

            uav_state.assign_to_cluster(
                cluster_id=cluster_state.cluster_id,
                status="en_route",
            )

            uav_state.move_towards(
                target_position=cluster_state.target_center,
                delta_time=delta_time,
                default_speed=self.config.default_uav_speed,
                current_time=current_time,
            )

    def _execute_strike_step(
        self,
        runtime_state: MissionRuntimeState,
        cluster_state: ClusterExecutionState,
        current_time: float,
    ) -> list[int]:
        """执行一个时间步内的打击动作。"""
        destroyed_ids: list[int] = []
        strike_count = 0

        while (
            cluster_state.current_order_index < len(cluster_state.ordered_target_ids)
            and strike_count < self.config.strike_targets_per_step
        ):
            target_id = int(
                cluster_state.ordered_target_ids[cluster_state.current_order_index]
            )
            cluster_state.current_order_index += 1

            if target_id in runtime_state.disappeared_target_ids:
                continue

            if target_id in runtime_state.destroyed_target_ids:
                continue

            try:
                runtime_state.mark_target_destroyed(
                    target_id=target_id,
                    event_time=current_time,
                )
            except Exception as exc:
                cluster_state.metadata.setdefault("strike_errors", []).append(
                    {
                        "target_id": target_id,
                        "error": str(exc),
                    }
                )
                continue

            destroyed_ids.append(target_id)
            strike_count += 1

        if cluster_state.current_order_index >= len(cluster_state.ordered_target_ids):
            cluster_state.status = "completed"
            cluster_state.completed_time = current_time

        return destroyed_ids

    def _get_cluster_assigned_uav_ids(
            self,
            cluster_state: ClusterExecutionState,
    ) -> list[int]:
        """
        获取某个 cluster 当前分配到的 UAV id。

        当前先使用 attack + guide。
        如果后续 ClusterExecutionState 增加 communication UAV，
        可以在这里继续扩展。
        """
        assigned_ids: list[int] = []

        assigned_ids.extend(int(value) for value in cluster_state.assigned_attack_uav_ids)
        assigned_ids.extend(int(value) for value in cluster_state.assigned_guide_uav_ids)

        return sorted(set(assigned_ids))

    def _update_cluster_position_from_uavs(
            self,
            runtime_state: MissionRuntimeState,
            cluster_state: ClusterExecutionState,
    ) -> None:
        """
        根据 cluster 内仍可用 UAV 的实时位置更新小组质心。

        cluster_state.current_position 现在只是用于可视化和摘要，
        真正运动状态以 UAVRuntimeState.current_position 为准。
        """
        assigned_uav_ids = self._get_cluster_assigned_uav_ids(cluster_state)

        positions = []

        for uav_id in assigned_uav_ids:
            if uav_id not in runtime_state.uav_runtime_states:
                continue

            uav_state = runtime_state.get_uav_runtime_state(uav_id)

            if uav_state.is_damaged:
                continue

            positions.append(uav_state.current_position)

        if not positions:
            return

        cluster_state.current_position = Position(
            x=sum(position.x for position in positions) / len(positions),
            y=sum(position.y for position in positions) / len(positions),
        )

    def _cluster_has_arrived(
            self,
            runtime_state: MissionRuntimeState,
            cluster_state: ClusterExecutionState,
    ) -> bool:
        """
        判断目标群对应 UAV 小组是否到达任务区域。

        当前规则：
        - 至少有一架可用攻击 UAV；
        - 所有可用攻击 UAV 到目标群中心的距离均小于 arrival_tolerance。

        后续可以改成：
        - 至少一架攻击 UAV 到达即可；
        - 或攻击 UAV + 侦察 UAV 都到达才算可打击。
        """
        active_attack_uavs = []

        for uav_id in cluster_state.assigned_attack_uav_ids:
            if uav_id not in runtime_state.uav_runtime_states:
                continue

            uav_state = runtime_state.get_uav_runtime_state(uav_id)

            if uav_state.is_damaged:
                continue

            active_attack_uavs.append(uav_state)

        if not active_attack_uavs:
            return False

        distances = [
            uav_state.distance_to(cluster_state.target_center)
            for uav_state in active_attack_uavs
        ]

        return max(distances) <= self.config.arrival_tolerance

    def _get_cluster_uav_position_summary(
            self,
            runtime_state: MissionRuntimeState,
            cluster_state: ClusterExecutionState,
    ) -> dict[int, list[float]]:
        """
        输出 cluster 内 UAV 的实时位置摘要，方便写入日志。
        """
        summary: dict[int, list[float]] = {}

        for uav_id in self._get_cluster_assigned_uav_ids(cluster_state):
            if uav_id not in runtime_state.uav_runtime_states:
                continue

            uav_state = runtime_state.get_uav_runtime_state(uav_id)

            summary[int(uav_id)] = [
                float(uav_state.current_position.x),
                float(uav_state.current_position.y),
            ]

        return summary

    def _handle_event(
        self,
        runtime_state: MissionRuntimeState,
        event: MissionEvent,
        original_battlefield_state,
    ) -> None:
        """应用动态事件并调用重规划控制器。"""
        self.logger.info(
            "Dynamic event triggered: "
            f"time={event.event_time}, "
            f"type={event.event_type.value}, "
            f"targets={event.affected_target_ids}, "
            f"uavs={event.affected_uav_ids}"
        )

        runtime_state.apply_event(event)

        replanning_result = self.replanning_controller.handle_event(
            runtime_state=runtime_state,
            event=event,
            original_battlefield_state=original_battlefield_state,
        )

        self.replanning_results.append(replanning_result)

        runtime_state.sync_assignments_from_mission_plan(runtime_state.mission_plan)

        self._sync_cluster_states_from_plan(
            mission_plan=runtime_state.mission_plan,
            runtime_state=runtime_state,
        )

    def _sync_cluster_states_from_plan(
            self,
            mission_plan: MissionPlan,
            runtime_state: MissionRuntimeState | None = None,
    ) -> None:
        """
        根据 MissionPlan 同步目标群执行状态。

        如果 runtime_state 不为空：
        - 新 cluster 的起点优先使用已分配 UAV 的实时位置均值；
        - 旧 cluster 保留当前执行进度；
        - UAV 的 assigned_cluster_id 会由 MissionRuntimeState 负责同步。
        """
        old_states = dict(self.cluster_states)
        new_states: dict[int, ClusterExecutionState] = {}

        for assignment in mission_plan.allocation_plan.assignments:
            cluster_id = int(assignment.cluster_id)
            strike_plan = mission_plan.strike_order_plans.get(cluster_id)

            if strike_plan is None:
                ordered_target_ids = assignment.target_cluster.target_ids
            else:
                ordered_target_ids = list(strike_plan.ordered_target_ids)

            if cluster_id in old_states:
                current_position = old_states[cluster_id].current_position
                status = old_states[cluster_id].status
                current_order_index = old_states[cluster_id].current_order_index
                reached_time = old_states[cluster_id].reached_time
                completed_time = old_states[cluster_id].completed_time
            else:
                current_position = self._resolve_initial_cluster_position(
                    assignment=assignment,
                    runtime_state=runtime_state,
                )
                status = "en_route"
                current_order_index = 0
                reached_time = None
                completed_time = None

            new_states[cluster_id] = ClusterExecutionState(
                cluster_id=cluster_id,
                current_position=current_position,
                target_center=assignment.target_cluster.center,
                assigned_attack_uav_ids=list(assignment.assigned_attack_uav_ids),
                assigned_guide_uav_ids=list(assignment.assigned_guide_uav_ids),
                ordered_target_ids=ordered_target_ids,
                status=status,
                current_order_index=current_order_index,
                reached_time=reached_time,
                completed_time=completed_time,
                metadata={
                    "synced_from_mission_plan": True,
                    "runtime_state_position_enabled": runtime_state is not None,
                },
            )

        self.cluster_states = new_states

    def _resolve_initial_cluster_position(
            self,
            assignment,
            runtime_state: MissionRuntimeState | None,
    ) -> Position:
        """
        解析新 cluster 的初始位置。

        优先级：
        1. 已分配 UAV 的实时位置均值；
        2. assignment.real_time_position；
        3. assignment.start_position；
        4. target_cluster.center。
        """
        if runtime_state is not None:
            assigned_uav_ids: list[int] = []

            assigned_uav_ids.extend(
                int(value) for value in assignment.assigned_attack_uav_ids
            )
            assigned_uav_ids.extend(
                int(value) for value in assignment.assigned_guide_uav_ids
            )

            if hasattr(assignment, "assigned_communication_uav_ids"):
                assigned_uav_ids.extend(
                    int(value)
                    for value in assignment.assigned_communication_uav_ids
                )

            positions = []

            for uav_id in assigned_uav_ids:
                if uav_id not in runtime_state.uav_runtime_states:
                    continue

                uav_state = runtime_state.get_uav_runtime_state(uav_id)

                if uav_state.is_damaged:
                    continue

                positions.append(uav_state.current_position)

            if positions:
                return Position(
                    x=sum(position.x for position in positions) / len(positions),
                    y=sum(position.y for position in positions) / len(positions),
                )

        return (
                assignment.real_time_position
                or assignment.start_position
                or assignment.target_cluster.center
        )

    def _build_step_record(
        self,
        runtime_state: MissionRuntimeState,
        current_time: float,
        events: list[MissionEvent],
        destroyed_this_step: list[int],
    ) -> SimulationStepRecord:
        """构造单步记录。"""
        completed_cluster_count = sum(
            1 for state in self.cluster_states.values() if state.is_completed
        )

        return SimulationStepRecord(
            time=current_time,
            event_types=[event.event_type.value for event in events],
            replanning_count=len(self.replanning_results),
            destroyed_target_ids=sorted(runtime_state.destroyed_target_ids),
            disappeared_target_ids=sorted(runtime_state.disappeared_target_ids),
            damaged_uav_ids=sorted(runtime_state.damaged_uav_ids),
            available_target_count=len(runtime_state.available_target_ids),
            available_uav_count=len(runtime_state.available_uav_ids),
            completed_cluster_count=completed_cluster_count,
            total_cluster_count=len(self.cluster_states),
            metadata={
                "destroyed_this_step": destroyed_this_step,
                "cluster_status": {
                    cluster_id: state.status
                    for cluster_id, state in self.cluster_states.items()
                },
                "cluster_positions": {
                    cluster_id: [
                        state.current_position.x,
                        state.current_position.y,
                    ]
                    for cluster_id, state in self.cluster_states.items()
                },
                "uav_status_counts": runtime_state.to_summary_dict().get(
                    "uav_status_counts",
                    {},
                ),
            },
        )

    def _build_uav_trajectory_records(
            self,
            runtime_state: MissionRuntimeState,
            current_time: float,
    ) -> list[UAVTrajectoryRecord]:
        """
        记录当前时间步所有 UAV 的运行时状态。

        用于验证：
        - UAV 是否随时间移动；
        - UAV 是否被分配到 cluster；
        - UAV 损毁后状态是否变成 damaged；
        - 重规划后 assigned_cluster_id 是否发生变化。
        """
        records: list[UAVTrajectoryRecord] = []

        for uav_id, uav_state in sorted(runtime_state.uav_runtime_states.items()):
            records.append(
                UAVTrajectoryRecord(
                    time=float(current_time),
                    uav_id=int(uav_id),
                    status=str(uav_state.status),
                    assigned_cluster_id=uav_state.assigned_cluster_id,
                    x=float(uav_state.current_position.x),
                    y=float(uav_state.current_position.y),
                    is_damaged=bool(uav_state.is_damaged),
                )
            )

        return records

    def _should_stop(self, runtime_state: MissionRuntimeState) -> bool:
        """判断仿真是否应该提前结束。"""
        if not self.config.stop_when_all_planned_targets_destroyed:
            return False

        planned_target_ids: set[int] = set()

        for plan in runtime_state.mission_plan.strike_order_plans.values():
            planned_target_ids.update(int(tid) for tid in plan.ordered_target_ids)

        if not planned_target_ids:
            return False

        unresolved_target_ids = planned_target_ids - runtime_state.destroyed_target_ids
        unresolved_target_ids -= runtime_state.disappeared_target_ids

        return len(unresolved_target_ids) == 0

    @staticmethod
    def _distance(position_a: Position, position_b: Position) -> float:
        return math.sqrt(
            (position_a.x - position_b.x) ** 2
            + (position_a.y - position_b.y) ** 2
        )


def load_mission_simulator_config(
    config: dict[str, Any],
) -> MissionSimulatorConfig:
    """从项目配置中读取 MissionSimulatorConfig。"""
    simulator_config = MissionSimulatorConfig(
        time_step=float(
            get_config_value(config, "scenario.time_step", default=30.0)
        ),
        max_time=float(
            get_config_value(config, "scenario.max_time", default=2200.0)
        ),
        default_uav_speed=float(
            get_config_value(
                config,
                "mission_simulator.default_uav_speed",
                default=80.0,
            )
        ),
        arrival_tolerance=float(
            get_config_value(
                config,
                "mission_simulator.arrival_tolerance",
                default=100.0,
            )
        ),
        strike_targets_per_step=int(
            get_config_value(
                config,
                "mission_simulator.strike_targets_per_step",
                default=1,
            )
        ),
        stop_when_all_planned_targets_destroyed=bool(
            get_config_value(
                config,
                "mission_simulator.stop_when_all_planned_targets_destroyed",
                default=True,
            )
        ),
        simulation_log_csv_path=str(
            get_config_value(
                config,
                "mission_simulator.output.simulation_log_csv_path",
                default="outputs/simulation/mission_simulation_log.csv",
            )
        ),
        uav_trajectory_csv_path=str(
            get_config_value(
                config,
                "mission_simulator.output.uav_trajectory_csv_path",
                default="outputs/simulation/uav_trajectory_log.csv",
            )
        ),
    )

    simulator_config.validate()
    return simulator_config