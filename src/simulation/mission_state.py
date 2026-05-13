from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from uav_dynamic_task_allocation.core.contracts import (
    MissionEvent,
    MissionEventType,
    MissionPlan,
)
from uav_dynamic_task_allocation.core.entities import Position, Target, UAV


class MissionStateError(Exception):
    """任务运行状态管理过程中的自定义错误。"""


@dataclass
class TargetRuntimeState:
    """
    单个目标的运行时状态。

    Target 是静态实体，描述目标自身属性：
        target_id, position, defense, significance, target_type 等。

    TargetRuntimeState 是动态状态，描述仿真执行过程中目标发生了什么：
        active / disappeared / destroyed
        disappeared_time / destroyed_time
        assigned_cluster_id

    这样做的目的：
        不直接修改原始 Target 实体；
        而是在运行时状态层记录目标是否仍然可用、是否已经被摧毁。
    """

    target: Target
    status: str = "active"
    assigned_cluster_id: int | None = None
    disappeared_time: float | None = None
    destroyed_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_id(self) -> int:
        return int(self.target.target_id)

    @property
    def is_available(self) -> bool:
        """
        当前目标是否仍可参与规划 / 打击。
        """
        return self.status == "active"

    @property
    def is_disappeared(self) -> bool:
        return self.status == "disappeared"

    @property
    def is_destroyed(self) -> bool:
        return self.status == "destroyed"

    def assign_to_cluster(self, cluster_id: int | None) -> None:
        """记录该目标当前被分配到哪个目标群。"""
        self.assigned_cluster_id = cluster_id

    def mark_disappeared(self, event_time: float) -> None:
        """
        标记目标消失。

        如果目标已经被摧毁，不再改成 disappeared，
        因为 destroyed 是更终态的结果。
        """
        if self.status == "destroyed":
            self.metadata.setdefault("ignored_disappear_after_destroyed", []).append(
                event_time
            )
            return

        self.status = "disappeared"
        self.disappeared_time = float(event_time)

    def mark_destroyed(self, event_time: float) -> None:
        """
        标记目标被摧毁。

        如果目标已经消失，则不允许再摧毁。
        """
        if self.status == "disappeared":
            raise MissionStateError(
                f"Cannot destroy disappeared target_id={self.target_id}"
            )

        self.status = "destroyed"
        self.destroyed_time = float(event_time)

    def reactivate(self, event_time: float) -> None:
        """
        目标重新出现。

        当前主要用于 target_appeared 事件中已有目标重新出现的情况。
        """
        self.status = "active"
        self.disappeared_time = None
        self.metadata.setdefault("reactivated_times", []).append(float(event_time))


@dataclass
class UAVRuntimeState:
    """
    单架 UAV 的运行时状态。

    UAV 是静态实体，描述 UAV 自身属性：
        uav_id, type, position, payload 等。

    UAVRuntimeState 是动态状态，描述仿真执行过程中的 UAV：
        current_position
        status
        assigned_cluster_id
        damaged_time
        remaining_payload
        last_update_time

    当前版本中，MissionSimulator 还会继续保留 cluster-level 执行状态。
    下一步会把 UAV 小组运动真正改成基于 UAVRuntimeState 更新。
    """

    uav: UAV
    current_position: Position
    status: str = "available"
    assigned_cluster_id: int | None = None
    damaged_time: float | None = None
    remaining_payload: float | None = None
    last_update_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_uav(
        cls,
        uav: UAV,
        current_time: float = 0.0,
    ) -> "UAVRuntimeState":
        """
        根据原始 UAV 实体初始化 UAV 运行时状态。
        """
        position = getattr(uav, "position", None)

        if position is None:
            position = Position(x=0.0, y=0.0)
        else:
            position = Position(x=float(position.x), y=float(position.y))

        remaining_payload = None

        for attr_name in ["payload", "load", "capacity", "attack_payload"]:
            if hasattr(uav, attr_name):
                try:
                    remaining_payload = float(getattr(uav, attr_name))
                    break
                except (TypeError, ValueError):
                    remaining_payload = None

        return cls(
            uav=uav,
            current_position=position,
            status="available",
            assigned_cluster_id=None,
            damaged_time=None,
            remaining_payload=remaining_payload,
            last_update_time=float(current_time),
            metadata={
                "source": "from_uav",
            },
        )

    @property
    def uav_id(self) -> int:
        return int(self.uav.uav_id)

    @property
    def is_available(self) -> bool:
        """
        UAV 是否可继续参与任务。

        damaged 视为不可用。
        assigned / en_route / striking 虽然已经被分配，
        但仍然是有效资源。
        """
        return self.status != "damaged"

    @property
    def is_damaged(self) -> bool:
        return self.status == "damaged"

    def get_speed(self, default_speed: float) -> float:
        """
        获取 UAV 速度。

        如果 UAV 实体中有 speed / velocity / max_speed 字段，则优先使用；
        否则使用 MissionSimulator 配置里的 default_uav_speed。
        """
        for attr_name in ["speed", "velocity", "max_speed", "cruise_speed"]:
            if hasattr(self.uav, attr_name):
                try:
                    value = float(getattr(self.uav, attr_name))
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass

        return float(default_speed)

    def assign_to_cluster(
        self,
        cluster_id: int | None,
        status: str = "assigned",
    ) -> None:
        """把 UAV 分配给某个目标群。"""
        if self.status == "damaged":
            return

        self.assigned_cluster_id = cluster_id
        self.status = status

    def mark_damaged(self, event_time: float) -> None:
        """标记 UAV 损毁。"""
        self.status = "damaged"
        self.damaged_time = float(event_time)

    def update_position(
        self,
        new_position: Position,
        current_time: float,
    ) -> None:
        """更新 UAV 实时位置。"""
        if self.status == "damaged":
            return

        self.current_position = Position(
            x=float(new_position.x),
            y=float(new_position.y),
        )
        self.last_update_time = float(current_time)

    def move_towards(
        self,
        target_position: Position,
        delta_time: float,
        default_speed: float,
        current_time: float,
    ) -> float:
        """
        按速度向目标位置移动。

        返回移动后的剩余距离。
        """
        if self.status == "damaged":
            return self.distance_to(target_position)

        distance = self.distance_to(target_position)

        if distance <= 1e-9:
            self.update_position(target_position, current_time)
            return 0.0

        speed = self.get_speed(default_speed)
        max_move_distance = speed * float(delta_time)

        if max_move_distance >= distance:
            self.update_position(target_position, current_time)
            return 0.0

        ratio = max_move_distance / distance

        new_position = Position(
            x=self.current_position.x
            + (target_position.x - self.current_position.x) * ratio,
            y=self.current_position.y
            + (target_position.y - self.current_position.y) * ratio,
        )

        self.update_position(new_position, current_time)

        return self.distance_to(target_position)

    def distance_to(self, target_position: Position) -> float:
        """计算当前 UAV 到某个位置的距离。"""
        return math.sqrt(
            (self.current_position.x - target_position.x) ** 2
            + (self.current_position.y - target_position.y) ** 2
        )


@dataclass
class MissionRuntimeState:
    """
    任务运行时状态。

    MissionPlan 描述的是“计划怎么做”；
    MissionRuntimeState 描述的是“任务执行到当前时刻，实际状态是什么”。

    当前版本同时保留：
        active_targets / active_uavs
        disappeared_target_ids / destroyed_target_ids / damaged_uav_ids

    这是为了兼容前面已经写好的 ReplanningController 和 MissionSimulator。

    但从现在开始，真正的动态状态以：
        target_runtime_states
        uav_runtime_states

    为主。
    """

    current_time: float
    mission_plan: MissionPlan

    active_targets: dict[int, Target]
    active_uavs: dict[int, UAV]

    target_runtime_states: dict[int, TargetRuntimeState] = field(
        default_factory=dict
    )
    uav_runtime_states: dict[int, UAVRuntimeState] = field(default_factory=dict)

    disappeared_target_ids: set[int] = field(default_factory=set)
    destroyed_target_ids: set[int] = field(default_factory=set)
    damaged_uav_ids: set[int] = field(default_factory=set)

    triggered_events: list[MissionEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_battlefield_state_and_plan(
        cls,
        battlefield_state,
        mission_plan: MissionPlan,
        current_time: float = 0.0,
    ) -> "MissionRuntimeState":
        """
        根据 BattlefieldState 和 MissionPlan 初始化运行时状态。

        这里会把最开始实例化得到的 UAV / Target 接入运行时状态：
            Target -> TargetRuntimeState
            UAV    -> UAVRuntimeState
        """
        active_targets = {
            int(target.target_id): target
            for target in battlefield_state.targets
        }

        active_uavs = {
            int(uav.uav_id): uav
            for uav in battlefield_state.uavs
        }

        target_runtime_states = {
            target_id: TargetRuntimeState(target=target)
            for target_id, target in active_targets.items()
        }

        uav_runtime_states = {
            uav_id: UAVRuntimeState.from_uav(
                uav=uav,
                current_time=current_time,
            )
            for uav_id, uav in active_uavs.items()
        }

        state = cls(
            current_time=float(current_time),
            mission_plan=mission_plan,
            active_targets=active_targets,
            active_uavs=active_uavs,
            target_runtime_states=target_runtime_states,
            uav_runtime_states=uav_runtime_states,
            metadata={
                "source": "from_battlefield_state_and_plan",
                "num_initial_targets": len(active_targets),
                "num_initial_uavs": len(active_uavs),
                "runtime_state_enabled": True,
            },
        )

        state.sync_assignments_from_mission_plan(mission_plan)
        state._sync_id_sets_from_runtime_states()
        state.validate()

        return state

    def validate(self) -> None:
        """检查运行时状态是否合法。"""
        if self.current_time < 0:
            raise MissionStateError("current_time must be non-negative.")

        duplicated_target_ids = self.disappeared_target_ids & self.destroyed_target_ids

        if duplicated_target_ids:
            raise MissionStateError(
                "A target cannot be both disappeared and destroyed: "
                f"{sorted(duplicated_target_ids)}"
            )

        missing_target_runtime_ids = set(self.active_targets) - set(
            self.target_runtime_states
        )
        if missing_target_runtime_ids:
            raise MissionStateError(
                "Missing TargetRuntimeState for target ids: "
                f"{sorted(missing_target_runtime_ids)}"
            )

        missing_uav_runtime_ids = set(self.active_uavs) - set(
            self.uav_runtime_states
        )
        if missing_uav_runtime_ids:
            raise MissionStateError(
                "Missing UAVRuntimeState for uav ids: "
                f"{sorted(missing_uav_runtime_ids)}"
            )

    @property
    def available_target_ids(self) -> list[int]:
        """
        当前仍可参与任务规划 / 打击的目标 ID。
        """
        return sorted(
            target_id
            for target_id, runtime_state in self.target_runtime_states.items()
            if runtime_state.is_available
        )

    @property
    def available_uav_ids(self) -> list[int]:
        """
        当前仍可用的 UAV ID。
        """
        return sorted(
            uav_id
            for uav_id, runtime_state in self.uav_runtime_states.items()
            if runtime_state.is_available
        )

    def get_available_targets(self) -> list[Target]:
        """返回当前仍可用目标对象。"""
        return [
            self.target_runtime_states[target_id].target
            for target_id in self.available_target_ids
        ]

    def get_available_uavs(self) -> list[UAV]:
        """返回当前仍可用 UAV 对象。"""
        return [
            self.uav_runtime_states[uav_id].uav
            for uav_id in self.available_uav_ids
        ]

    def get_target_runtime_state(self, target_id: int) -> TargetRuntimeState:
        """读取某个目标的运行时状态。"""
        target_id = int(target_id)

        if target_id not in self.target_runtime_states:
            raise MissionStateError(f"Unknown target_id={target_id}")

        return self.target_runtime_states[target_id]

    def get_uav_runtime_state(self, uav_id: int) -> UAVRuntimeState:
        """读取某架 UAV 的运行时状态。"""
        uav_id = int(uav_id)

        if uav_id not in self.uav_runtime_states:
            raise MissionStateError(f"Unknown uav_id={uav_id}")

        return self.uav_runtime_states[uav_id]

    def get_cluster_uav_runtime_states(
        self,
        cluster_id: int,
    ) -> list[UAVRuntimeState]:
        """
        返回分配给某个目标群的 UAV 运行时状态。
        """
        return [
            runtime_state
            for runtime_state in self.uav_runtime_states.values()
            if runtime_state.assigned_cluster_id == int(cluster_id)
        ]

    def get_cluster_target_runtime_states(
        self,
        cluster_id: int,
    ) -> list[TargetRuntimeState]:
        """
        返回分配给某个目标群的目标运行时状态。
        """
        return [
            runtime_state
            for runtime_state in self.target_runtime_states.values()
            if runtime_state.assigned_cluster_id == int(cluster_id)
        ]

    def sync_assignments_from_mission_plan(
        self,
        mission_plan: MissionPlan | None = None,
    ) -> None:
        """
        根据 MissionPlan 同步 UAV / Target 的 assigned_cluster_id。

        每次 MissionPlan 重规划之后都应该调用一次。
        后续 MissionSimulator 在重规划后会使用这个函数。
        """
        mission_plan = mission_plan or self.mission_plan

        # 先清空旧 assignment，但不改变 destroyed / damaged 等终态。
        for target_state in self.target_runtime_states.values():
            target_state.assign_to_cluster(None)

        for uav_state in self.uav_runtime_states.values():
            if not uav_state.is_damaged:
                uav_state.assign_to_cluster(None, status="available")

        for assignment in mission_plan.allocation_plan.assignments:
            cluster_id = int(assignment.cluster_id)

            for target in assignment.target_cluster.targets:
                target_id = int(target.target_id)

                if target_id in self.target_runtime_states:
                    self.target_runtime_states[target_id].assign_to_cluster(cluster_id)

            assigned_uav_ids = []
            assigned_uav_ids.extend(int(value) for value in assignment.assigned_attack_uav_ids)
            assigned_uav_ids.extend(int(value) for value in assignment.assigned_guide_uav_ids)

            if hasattr(assignment, "assigned_communication_uav_ids"):
                assigned_uav_ids.extend(
                    int(value) for value in assignment.assigned_communication_uav_ids
                )

            for uav_id in assigned_uav_ids:
                if uav_id not in self.uav_runtime_states:
                    continue

                self.uav_runtime_states[uav_id].assign_to_cluster(
                    cluster_id=cluster_id,
                    status="assigned",
                )

        self._sync_id_sets_from_runtime_states()
        self.validate()

    def apply_event(self, event: MissionEvent) -> None:
        """
        将动态事件应用到运行时状态。

        当前只更新状态。
        重规划由 ReplanningController 负责。
        """
        event_type = event.event_type

        if event_type == MissionEventType.TARGET_DISAPPEARED:
            self._mark_targets_disappeared(
                target_ids=event.affected_target_ids,
                event_time=event.event_time,
            )

        elif event_type == MissionEventType.TARGET_APPEARED:
            self._record_target_appeared(event)

        elif event_type in {
            MissionEventType.ATTACK_UAV_DESTROYED,
            MissionEventType.GUIDE_UAV_DESTROYED,
            MissionEventType.COMMUNICATION_UAV_DESTROYED,
        }:
            self._mark_uavs_damaged(
                uav_ids=event.affected_uav_ids,
                event_time=event.event_time,
            )

        elif event_type == MissionEventType.RESOURCE_SHORTAGE:
            self.metadata.setdefault("resource_shortage_events", []).append(
                self._event_to_dict(event)
            )

        elif event_type == MissionEventType.REPLANNING_TRIGGERED:
            self.metadata.setdefault("replanning_triggered_events", []).append(
                self._event_to_dict(event)
            )

        elif event_type == MissionEventType.CUSTOM:
            self.metadata.setdefault("custom_events", []).append(
                self._event_to_dict(event)
            )

        else:
            raise MissionStateError(f"Unsupported event type: {event_type}")

        self.triggered_events.append(event)
        self.current_time = max(self.current_time, float(event.event_time))

        self._sync_id_sets_from_runtime_states()
        self.validate()

    def mark_target_destroyed(
        self,
        target_id: int,
        event_time: float | None = None,
    ) -> None:
        """
        标记目标被摧毁。

        MissionSimulator 在执行打击动作后会调用这个函数。
        """
        target_id = int(target_id)
        event_time = self.current_time if event_time is None else float(event_time)

        target_state = self.get_target_runtime_state(target_id)
        target_state.mark_destroyed(event_time)

        self._sync_id_sets_from_runtime_states()
        self.validate()

    def mark_uav_damaged(
        self,
        uav_id: int,
        event_time: float | None = None,
    ) -> None:
        """
        标记 UAV 损毁。
        """
        uav_id = int(uav_id)
        event_time = self.current_time if event_time is None else float(event_time)

        uav_state = self.get_uav_runtime_state(uav_id)
        uav_state.mark_damaged(event_time)

        self._sync_id_sets_from_runtime_states()
        self.validate()

    def update_uav_position(
        self,
        uav_id: int,
        new_position: Position,
        current_time: float,
    ) -> None:
        """
        更新单架 UAV 的实时位置。
        """
        uav_state = self.get_uav_runtime_state(uav_id)
        uav_state.update_position(
            new_position=new_position,
            current_time=current_time,
        )

    def to_summary_dict(self) -> dict[str, Any]:
        """输出当前任务状态摘要。"""
        target_status_counts = self._count_status(
            state.status for state in self.target_runtime_states.values()
        )
        uav_status_counts = self._count_status(
            state.status for state in self.uav_runtime_states.values()
        )

        return {
            "current_time": self.current_time,
            "num_active_targets_total": len(self.active_targets),
            "num_available_targets": len(self.available_target_ids),
            "num_disappeared_targets": len(self.disappeared_target_ids),
            "num_destroyed_targets": len(self.destroyed_target_ids),
            "num_active_uavs_total": len(self.active_uavs),
            "num_available_uavs": len(self.available_uav_ids),
            "num_damaged_uavs": len(self.damaged_uav_ids),
            "available_target_ids": self.available_target_ids,
            "disappeared_target_ids": sorted(self.disappeared_target_ids),
            "destroyed_target_ids": sorted(self.destroyed_target_ids),
            "available_uav_ids": self.available_uav_ids,
            "damaged_uav_ids": sorted(self.damaged_uav_ids),
            "target_status_counts": target_status_counts,
            "uav_status_counts": uav_status_counts,
            "num_triggered_events": len(self.triggered_events),
        }

    def _mark_targets_disappeared(
        self,
        target_ids: list[int],
        event_time: float,
    ) -> None:
        """标记目标消失。"""
        for target_id in target_ids:
            target_id = int(target_id)

            if target_id not in self.target_runtime_states:
                self.metadata.setdefault("unknown_disappeared_target_ids", []).append(
                    target_id
                )
                continue

            self.target_runtime_states[target_id].mark_disappeared(event_time)

    def _record_target_appeared(self, event: MissionEvent) -> None:
        """
        记录新目标出现。

        当前阶段：
        - 如果 target_id 已经存在，则重新激活；
        - 如果 target_id 不存在，只记录 metadata。

        后续可以增加 DynamicTargetLoader，
        从 new_target_source 或外部数据中构造新的 Target 实体。
        """
        for target_id in event.affected_target_ids:
            target_id = int(target_id)

            if target_id in self.target_runtime_states:
                self.target_runtime_states[target_id].reactivate(event.event_time)
                continue

            target = self._build_target_from_event_metadata(target_id, event)
            if target is not None:
                self.active_targets[target_id] = target
                self.target_runtime_states[target_id] = TargetRuntimeState(
                    target=target,
                    status="active",
                    metadata={
                        "created_by_event": True,
                        "event_time": event.event_time,
                    },
                )
                self.metadata.setdefault("created_target_ids", []).append(target_id)
                continue

            self.metadata.setdefault("appeared_unknown_target_ids", []).append(
                target_id
            )
            self.metadata.setdefault("target_appeared_events", []).append(
                self._event_to_dict(event)
            )

    def _build_target_from_event_metadata(
        self,
        target_id: int,
        event: MissionEvent,
    ) -> Target | None:
        """Create a new target from target_appeared event metadata when provided."""
        candidates = event.metadata.get("new_targets", [])
        if isinstance(candidates, dict):
            candidates = [candidates]

        for raw_target in candidates:
            if int(raw_target.get("target_id", target_id)) != int(target_id):
                continue
            try:
                return Target(
                    target_id=int(target_id),
                    position=Position(
                        x=float(raw_target["x"]),
                        y=float(raw_target["y"]),
                    ),
                    target_type=int(raw_target.get("target_type", 1)),
                    defense=float(raw_target.get("defense", 8.0)),
                    significance=float(raw_target.get("significance", 10.0)),
                    metadata={
                        "source": "target_appeared_event",
                        "event_time": event.event_time,
                    },
                )
            except KeyError as exc:
                self.metadata.setdefault("invalid_new_target_payloads", []).append(
                    {
                        "target_id": target_id,
                        "missing_key": str(exc),
                        "payload": raw_target,
                    }
                )
                return None
        return None

    def _mark_uavs_damaged(
        self,
        uav_ids: list[int],
        event_time: float,
    ) -> None:
        """标记 UAV 损毁。"""
        for uav_id in uav_ids:
            uav_id = int(uav_id)

            if uav_id not in self.uav_runtime_states:
                self.metadata.setdefault("unknown_damaged_uav_ids", []).append(uav_id)
                continue

            self.uav_runtime_states[uav_id].mark_damaged(event_time)

    def _sync_id_sets_from_runtime_states(self) -> None:
        """
        根据 runtime state 同步旧版 ID 集合。

        这样可以兼容前面已经写好的：
            disappeared_target_ids
            destroyed_target_ids
            damaged_uav_ids
        """
        self.disappeared_target_ids = {
            target_id
            for target_id, state in self.target_runtime_states.items()
            if state.is_disappeared
        }

        self.destroyed_target_ids = {
            target_id
            for target_id, state in self.target_runtime_states.items()
            if state.is_destroyed
        }

        self.damaged_uav_ids = {
            uav_id
            for uav_id, state in self.uav_runtime_states.items()
            if state.is_damaged
        }

    @staticmethod
    def _event_to_dict(event: MissionEvent) -> dict[str, Any]:
        """将 MissionEvent 转成字典，方便写入 metadata。"""
        return {
            "event_time": event.event_time,
            "event_type": event.event_type.value,
            "affected_target_ids": list(event.affected_target_ids),
            "affected_uav_ids": list(event.affected_uav_ids),
            "metadata": dict(event.metadata),
        }

    @staticmethod
    def _count_status(status_values) -> dict[str, int]:
        """统计状态数量。"""
        counts: dict[str, int] = {}

        for status in status_values:
            counts[status] = counts.get(status, 0) + 1

        return counts
