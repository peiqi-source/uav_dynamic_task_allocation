from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from uav_dynamic_task_allocation.core.contracts import (
    MissionEvent,
    MissionEventType,
)
from uav_dynamic_task_allocation.utils.config import get_config_value


class DynamicEventError(Exception):
    """动态事件配置、解析和触发过程中的自定义错误。"""


@dataclass(frozen=True)
class DynamicEventScheduleConfig:
    """
    动态事件时间表配置。

    enabled:
        是否启用动态事件。

    event_schedule:
        从 YAML 中读取的事件列表。

    trigger_tolerance:
        判断某个事件是否在当前时间步触发的容差。
        例如 time_step=30 时，事件时间最好刚好落在 30 的倍数；
        如果不是，也可以用容差处理。
    """

    enabled: bool = False
    event_schedule: list[dict[str, Any]] = field(default_factory=list)
    trigger_tolerance: float = 1e-6

    def validate(self) -> None:
        """检查配置是否合法。"""
        if self.trigger_tolerance < 0:
            raise DynamicEventError("trigger_tolerance must be non-negative.")

        if not isinstance(self.event_schedule, list):
            raise DynamicEventError("event_schedule must be a list.")


class DynamicEventManager:
    """
    动态事件管理器。

    它只负责：
    1. 从配置中读取动态事件；
    2. 在给定时间点返回应该触发的事件；
    3. 记录哪些事件已经触发过。

    它不负责重规划。
    后续 ReplanningController 会根据事件类型决定重规划范围。
    """

    def __init__(self, config: DynamicEventScheduleConfig) -> None:
        config.validate()

        self.config = config
        self.events = self._parse_events(config.event_schedule)
        self.triggered_event_indices: set[int] = set()

    def get_events_at_time(self, current_time: float) -> list[MissionEvent]:
        """
        返回当前时间应该触发的事件。

        如果 dynamic_events.enabled=false，则永远返回空列表。
        """
        if not self.config.enabled:
            return []

        triggered_events: list[MissionEvent] = []

        for index, event in enumerate(self.events):
            if index in self.triggered_event_indices:
                continue

            if abs(float(event.event_time) - float(current_time)) <= (
                self.config.trigger_tolerance
            ):
                triggered_events.append(event)
                self.triggered_event_indices.add(index)

        return triggered_events

    def get_events_between(
        self,
        previous_time: float,
        current_time: float,
    ) -> list[MissionEvent]:
        """
        返回 (previous_time, current_time] 之间应该触发的事件。

        这个函数比 get_events_at_time 更适合 MissionSimulator，
        因为时间步推进可能跨过某个事件时间。
        """
        if not self.config.enabled:
            return []

        if current_time < previous_time:
            raise DynamicEventError("current_time must be >= previous_time.")

        triggered_events: list[MissionEvent] = []

        for index, event in enumerate(self.events):
            if index in self.triggered_event_indices:
                continue

            event_time = float(event.event_time)

            if previous_time < event_time <= current_time:
                triggered_events.append(event)
                self.triggered_event_indices.add(index)

        return triggered_events

    def reset(self) -> None:
        """重置已触发事件记录。"""
        self.triggered_event_indices.clear()

    def remaining_events(self) -> list[MissionEvent]:
        """返回尚未触发的事件。"""
        return [
            event
            for index, event in enumerate(self.events)
            if index not in self.triggered_event_indices
        ]

    def _parse_events(
        self,
        raw_events: list[dict[str, Any]],
    ) -> list[MissionEvent]:
        """解析 YAML 中的事件配置。"""
        events: list[MissionEvent] = []

        for index, raw_event in enumerate(raw_events):
            try:
                event = self._parse_single_event(raw_event)
            except Exception as exc:
                raise DynamicEventError(
                    f"Failed to parse dynamic event at index={index}: {raw_event}"
                ) from exc

            events.append(event)

        events.sort(key=lambda item: float(item.event_time))
        return events

    def _parse_single_event(
        self,
        raw_event: dict[str, Any],
    ) -> MissionEvent:
        """解析单个事件配置。"""
        if "time" not in raw_event:
            raise DynamicEventError(f"Event missing required field: time. {raw_event}")

        if "type" not in raw_event:
            raise DynamicEventError(f"Event missing required field: type. {raw_event}")

        event_time = float(raw_event["time"])
        event_type = self._parse_event_type(str(raw_event["type"]))

        affected_target_ids = [
            int(value)
            for value in raw_event.get("affected_target_ids", [])
        ]

        affected_uav_ids = [
            int(value)
            for value in raw_event.get("affected_uav_ids", [])
        ]

        metadata = {
            key: value
            for key, value in raw_event.items()
            if key
            not in {
                "time",
                "type",
                "affected_target_ids",
                "affected_uav_ids",
            }
        }

        return MissionEvent(
            event_time=event_time,
            event_type=event_type,
            affected_target_ids=affected_target_ids,
            affected_uav_ids=affected_uav_ids,
            metadata=metadata,
        )

    @staticmethod
    def _parse_event_type(event_type: str) -> MissionEventType:
        """解析事件类型字符串。"""
        normalized = event_type.lower()

        try:
            return MissionEventType(normalized)
        except ValueError as exc:
            raise DynamicEventError(
                f"Unsupported event type: {event_type}"
            ) from exc


def load_dynamic_event_schedule_config(
    config: dict[str, Any],
) -> DynamicEventScheduleConfig:
    """从项目总配置中读取动态事件配置。"""
    event_config = DynamicEventScheduleConfig(
        enabled=bool(
            get_config_value(
                config,
                "scenario.dynamic_events.enabled",
                default=False,
            )
        ),
        event_schedule=list(
            get_config_value(
                config,
                "scenario.dynamic_events.event_schedule",
                default=[],
            )
        ),
        trigger_tolerance=float(
            get_config_value(
                config,
                "scenario.dynamic_events.trigger_tolerance",
                default=1e-6,
            )
        ),
    )

    event_config.validate()
    return event_config


def build_dynamic_event_manager(
    config: dict[str, Any],
) -> DynamicEventManager:
    """根据项目总配置构造 DynamicEventManager。"""
    return DynamicEventManager(
        config=load_dynamic_event_schedule_config(config)
    )