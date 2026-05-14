"""数据模块中的schemas 数据实现。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataSchema:
    """
    Define a standard data schema.

    Attributes:
        required_columns: Columns that must exist after standardization.
        numeric_columns: Columns that should be numeric.
        column_aliases: Mapping from raw column names to standard column names.
    """

    # required_columns: 标准化后必须存在的列名列表。
    required_columns: list[str]
    # numeric_columns: 需要转换为数值类型的列名列表。
    numeric_columns: list[str]
    # column_aliases: 原始 CSV 列名到标准列名的映射关系。
    column_aliases: dict[str, str]


UAV_SCHEMA = DataSchema(
    required_columns=[
        "uav_id",
        "x",
        "y",
        "uav_type",
        "work_range",
        "attack_power",
    ],
    numeric_columns=[
        "uav_id",
        "x",
        "y",
        "uav_type",
        "work_range",
        "attack_power",
    ],
    column_aliases={
        "序号": "uav_id",
        "x": "x",
        "y": "y",
        "type": "uav_type",
        "无人机类型": "uav_type",
        "无人机类型（1导引、2通信、3攻击）": "uav_type",
        "工作距离": "work_range",
        "攻击能力": "attack_power",
    },
)


TARGET_SCHEMA = DataSchema(
    required_columns=[
        "target_id",
        "x",
        "y",
        "target_type",
        "defense",
        "significance",
    ],
    numeric_columns=[
        "target_id",
        "x",
        "y",
        "target_type",
        "defense",
        "significance",
    ],
    column_aliases={
        "nun": "target_id",
        "num": "target_id",
        "序号": "target_id",
        "x": "x",
        "y": "y",
        "位置x": "x",
        "位置y": "y",
        "type": "target_type",
        "类型type": "target_type",
        "defense": "defense",
        "防御力defense": "defense",
        "significance": "significance",
        "重要性significance": "significance",
        "图显类型type": "display_type",
    },
)
