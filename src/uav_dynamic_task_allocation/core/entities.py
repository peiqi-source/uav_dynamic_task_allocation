"""core 数据模块中的entities 数据实现。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from typing import Any

import pandas as pd


class EntityError(Exception):
    """核心实体构建过程中的自定义错误。"""


class UAVType(Enum):
    """无人机类型定义。"""

    # GUIDE: 导引。
    GUIDE = 1
    # COMMUNICATION: 通信。
    COMMUNICATION = 2
    # ATTACK: 攻击。
    ATTACK = 3

    @classmethod
    def from_value(cls, value: int | float) -> "UAVType":
        """根据原始数值生成无人机类型。"""
        value = int(value)

        for item in cls:
            if item.value == value:
                return item

        raise EntityError(
            f"Invalid UAV type: {value}. "
            "Valid values are: 1=GUIDE, 2=COMMUNICATION, 3=ATTACK."
        )


class EntityStatus(Enum):
    """实体状态定义。"""

    # ACTIVE: 可用状态。
    ACTIVE = "active"
    # INACTIVE: 不可用状态。
    INACTIVE = "inactive"
    # DESTROYED: 毁伤状态。
    DESTROYED = "destroyed"


@dataclass
class Position:
    """二维战场坐标。"""

    # x: 横坐标。
    x: float
    # y: 纵坐标。
    y: float

    def distance_to(self, other: "Position") -> float:
        """计算当前坐标到另一个坐标的欧氏距离。"""
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class UAV:
    """
    无人机实体。

    该类用于统一描述无人机的基础属性，为后续任务分配、
    路径规划、强化学习环境构建提供标准对象。
    """

    # uav_id: 无人机编号。
    uav_id: int
    # position: 位置坐标。
    position: Position
    # uav_type: 无人机类型。
    uav_type: UAVType
    # work_range: 无人机工作半径。
    work_range: float
    # attack_power: 无人机攻击能力。
    attack_power: float
    # status: 状态。
    status: EntityStatus = EntityStatus.ACTIVE
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """判断无人机是否处于可用状态。"""
        return self.status == EntityStatus.ACTIVE

    @property
    def is_attack_uav(self) -> bool:
        """判断是否为攻击无人机。"""
        return self.uav_type == UAVType.ATTACK

    @property
    def is_guide_uav(self) -> bool:
        """判断是否为导引无人机。"""
        return self.uav_type == UAVType.GUIDE

    @property
    def is_communication_uav(self) -> bool:
        """判断是否为通信无人机。"""
        return self.uav_type == UAVType.COMMUNICATION

    def distance_to_target(self, target: "Target") -> float:
        """计算无人机到目标的距离。"""
        return self.position.distance_to(target.position)

    def can_reach(self, target: "Target") -> bool:
        """
        判断无人机是否在工作距离内可以覆盖目标。

        注意：这里只判断距离约束，不判断任务类型、弹药、协同约束等。
        """
        return self.distance_to_target(target) <= self.work_range

    @classmethod
    def from_series(cls, row: pd.Series) -> "UAV":
        """从 DataFrame 的一行数据构建 UAV 对象。"""
        return cls(
            uav_id=int(row["uav_id"]),
            position=Position(
                x=float(row["x"]),
                y=float(row["y"]),
            ),
            uav_type=UAVType.from_value(row["uav_type"]),
            work_range=float(row["work_range"]),
            attack_power=float(row["attack_power"]),
        )


@dataclass
class Target:
    """
    目标实体。

    该类用于描述战场目标的基础属性，包括位置、类型、防御力和重要性。
    """

    # target_id: 目标编号。
    target_id: int
    # position: 位置坐标。
    position: Position
    # target_type: 目标类型。
    target_type: int
    # defense: 防御能力。
    defense: float
    # significance: 重要程度。
    significance: float
    # status: 状态。
    status: EntityStatus = EntityStatus.ACTIVE
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """判断目标是否仍然有效。"""
        return self.status == EntityStatus.ACTIVE

    @property
    def value_score(self) -> float:
        """
        目标价值评分。

        当前先使用重要性作为目标价值。后续可以扩展为：
        significance、defense、target_type、威胁程度等因素的综合评分。
        """
        return self.significance

    @classmethod
    def from_series(cls, row: pd.Series) -> "Target":
        """从 DataFrame 的一行数据构建 Target 对象。"""
        metadata = {}

        if "display_type" in row.index:
            metadata["display_type"] = row["display_type"]

        return cls(
            target_id=int(row["target_id"]),
            position=Position(
                x=float(row["x"]),
                y=float(row["y"]),
            ),
            target_type=int(row["target_type"]),
            defense=float(row["defense"]),
            significance=float(row["significance"]),
            metadata=metadata,
        )


@dataclass
class BattlefieldState:
    """
    战场状态对象。

    该类用于统一维护当前仿真时刻的无人机集合、目标集合和步骤信息。
    后续环境 reset()、step()、动态事件处理都会围绕该对象展开。
    """

    # uavs: uavs 数据。
    uavs: list[UAV]
    # targets: 目标集合。
    targets: list[Target]
    # current_step: 当前步数。
    current_step: int = 0
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def active_uavs(self) -> list[UAV]:
        """返回当前仍然可用的无人机。"""
        return [uav for uav in self.uavs if uav.is_active]

    @property
    def active_targets(self) -> list[Target]:
        """返回当前仍然有效的目标。"""
        return [target for target in self.targets if target.is_active]

    @property
    def attack_uavs(self) -> list[UAV]:
        """返回当前可用的攻击无人机。"""
        return [
            uav for uav in self.active_uavs
            if uav.uav_type == UAVType.ATTACK
        ]

    @property
    def guide_uavs(self) -> list[UAV]:
        """返回当前可用的导引无人机。"""
        return [
            uav for uav in self.active_uavs
            if uav.uav_type == UAVType.GUIDE
        ]

    @property
    def communication_uavs(self) -> list[UAV]:
        """返回当前可用的通信无人机。"""
        return [
            uav for uav in self.active_uavs
            if uav.uav_type == UAVType.COMMUNICATION
        ]

    def get_uav_by_id(self, uav_id: int) -> UAV:
        """根据编号查找无人机。"""
        for uav in self.uavs:
            if uav.uav_id == uav_id:
                return uav

        raise EntityError(f"UAV not found: uav_id={uav_id}")

    def get_target_by_id(self, target_id: int) -> Target:
        """根据编号查找目标。"""
        for target in self.targets:
            if target.target_id == target_id:
                return target

        raise EntityError(f"Target not found: target_id={target_id}")

    def summary(self) -> dict[str, int]:
        """返回当前战场状态的基础统计信息。"""
        return {
            "total_uavs": len(self.uavs),
            "active_uavs": len(self.active_uavs),
            "guide_uavs": len(self.guide_uavs),
            "communication_uavs": len(self.communication_uavs),
            "attack_uavs": len(self.attack_uavs),
            "total_targets": len(self.targets),
            "active_targets": len(self.active_targets),
            "current_step": self.current_step,
        }


def build_uavs_from_dataframe(df: pd.DataFrame) -> list[UAV]:
    """
    将标准化 UAV DataFrame 转换为 UAV 对象列表。

    输入 DataFrame 必须已经经过 loaders.py 和 validators.py 处理。
    """
    return [UAV.from_series(row) for _, row in df.iterrows()]


def build_targets_from_dataframe(df: pd.DataFrame) -> list[Target]:
    """
    将标准化 Target DataFrame 转换为 Target 对象列表。

    输入 DataFrame 必须已经经过 loaders.py 和 validators.py 处理。
    """
    return [Target.from_series(row) for _, row in df.iterrows()]


def build_battlefield_state(
    uav_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> BattlefieldState:
    """
    根据 UAV 和 Target 数据构建初始战场状态。
    """
    uavs = build_uavs_from_dataframe(uav_df)
    targets = build_targets_from_dataframe(target_df)

    return BattlefieldState(
        uavs=uavs,
        targets=targets,
        current_step=0,
    )