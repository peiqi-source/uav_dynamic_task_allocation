from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any

import pandas as pd

from uav_dynamic_task_allocation.core.entities import Position
from uav_dynamic_task_allocation.utils.config import get_config_value


class EnvConfigError(Exception):
    """环境配置解析和校验过程中的自定义错误。"""


@dataclass
class BattlefieldBounds:
    """
    二维战场边界。

    该对象用于统一描述仿真环境中的空间范围。这里的边界不是简单的绘图参数，
    而是后续动作合法性判断、位置归一化、可视化范围、路径规划和环境观测构造
    的基础。

    当前项目中同时存在两类坐标尺度：
    1. 原始 drone_battle_env.py 中 0~100 的 toy 坐标；
    2. main.py、CSV 数据和论文仿真中使用的战场级坐标。
    因此边界不能写死在环境代码里，必须由配置文件提供，并允许根据真实数据自动扩展。
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def validate(self) -> None:
        """检查边界是否合法，避免出现 x_min >= x_max 这类基础配置错误。"""
        if self.x_min >= self.x_max:
            raise EnvConfigError(
                f"Invalid x bounds: x_min={self.x_min}, x_max={self.x_max}"
            )

        if self.y_min >= self.y_max:
            raise EnvConfigError(
                f"Invalid y bounds: y_min={self.y_min}, y_max={self.y_max}"
            )

    def contains(self, position: Position) -> bool:
        """
        判断某个位置是否位于战场边界内。

        这个函数后面会被用于：
        1. 检查 UAV / Target 初始数据是否越界；
        2. 判断无人机移动后是否飞出作战区域；
        3. 构造强化学习环境中的非法动作惩罚。
        """
        return (
            self.x_min <= position.x <= self.x_max
            and self.y_min <= position.y <= self.y_max
        )

    def expanded_by_dataframe(
        self,
        dataframes: list[pd.DataFrame],
        margin_ratio: float = 0.05,
    ) -> "BattlefieldBounds":
        """
        根据已有 UAV / Target 数据自动扩展战场边界。

        这样设计是为了兼容旧代码和真实数据：
        有些旧脚本中的坐标范围并没有统一写在一个配置文件里，
        如果我们只依赖手动配置，容易因为边界设置过窄导致有效数据被误判为越界。
        因此这里允许环境根据数据自动扩展边界，同时保留配置文件中的边界作为初始参考。
        """
        x_values: list[float] = []
        y_values: list[float] = []

        for df in dataframes:
            if "x" in df.columns:
                x_values.extend(df["x"].dropna().astype(float).tolist())
            if "y" in df.columns:
                y_values.extend(df["y"].dropna().astype(float).tolist())

        if not x_values or not y_values:
            return self

        data_x_min = min(x_values)
        data_x_max = max(x_values)
        data_y_min = min(y_values)
        data_y_max = max(y_values)

        x_span = max(data_x_max - data_x_min, 1.0)
        y_span = max(data_y_max - data_y_min, 1.0)

        x_margin = x_span * margin_ratio
        y_margin = y_span * margin_ratio

        return BattlefieldBounds(
            x_min=min(self.x_min, data_x_min - x_margin),
            x_max=max(self.x_max, data_x_max + x_margin),
            y_min=min(self.y_min, data_y_min - y_margin),
            y_max=max(self.y_max, data_y_max + y_margin),
        )


@dataclass
class DynamicEventSchedule:
    """
    确定性动态事件时间表。

    原始 main.py 中使用 if t == 300、if t == 600 这种方式触发事件。
    在新工程里不建议把这些时间点硬编码在环境主循环中，而是先配置化，
    后续交给 events/event_manager.py 统一调度。
    """

    target_disappear_times: list[float] = field(default_factory=list)
    target_appear_times: list[float] = field(default_factory=list)
    attack_uav_destroyed_times: list[float] = field(default_factory=list)
    guide_uav_destroyed_times: list[float] = field(default_factory=list)


@dataclass
class RandomEventProbabilities:
    """
    随机动态事件概率。

    原始 drone_battle_env.py 中包含随机目标出现、目标消失和无人机损毁的简化逻辑。
    这些概率在新工程中不直接写进 step()，而是先作为环境参数保存，
    后续由事件模块根据 enable_dynamic_events 决定是否使用。
    """

    target_appear: float = 0.0
    target_disappear: float = 0.0
    uav_destroyed: float = 0.0

    def validate(self) -> None:
        """检查概率是否位于 [0, 1] 区间。"""
        for name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise EnvConfigError(
                    f"Invalid random event probability: {name}={value}. "
                    "Probability must be in [0, 1]."
                )


@dataclass
class EnvConfig:
    """
    UAV 对地目标任务决策环境配置。

    该类是 configs/default.yaml 中 env 部分的 Python 表达。
    这样做的目的不是多写一层代码，而是把“松散字典”转换成“有类型、有校验、
    有默认值、有业务含义”的配置对象，减少后续环境模块中的硬编码和 KeyError。
    """

    name: str = "drone_battle"

    coordinate_mode: str = "battlefield"
    coordinate_unit: str = "m"
    bounds: BattlefieldBounds = field(
        default_factory=lambda: BattlefieldBounds(
            x_min=-2000.0,
            x_max=40000.0,
            y_min=-170000.0,
            y_max=2000.0,
        )
    )
    auto_infer_bounds_from_data: bool = True
    boundary_margin_ratio: float = 0.05

    max_time: float = 2200.0
    time_step: float = 30.0
    uav_speed: float = 70.0

    base_position: Position = field(
        default_factory=lambda: Position(x=0.0, y=-16000.0)
    )

    attack_payload_per_uav: float = 6.0
    communication_covers_battlefield: bool = True

    enable_dynamic_events: bool = False
    event_schedule: DynamicEventSchedule = field(
        default_factory=DynamicEventSchedule
    )
    random_event_probabilities: RandomEventProbabilities = field(
        default_factory=RandomEventProbabilities
    )

    @property
    def max_steps(self) -> int:
        """
        根据最大仿真时间和时间步长计算 episode 最大步数。

        原始 main.py 的逻辑是：
            t = 0
            while t < 2200:
                ...
                t += 30

        在强化学习环境中，我们通常用 current_step 表示推进次数，
        因此这里将 max_time / time_step 转换成 max_steps。
        """
        return int(ceil(self.max_time / self.time_step))

    def validate(self) -> None:
        """
        对环境配置进行基础合法性检查。

        这些检查放在环境初始化前完成，能够尽早暴露配置错误，
        避免模型训练到一半才发现 max_time、time_step 或事件概率有问题。
        """
        if self.coordinate_mode not in {"toy", "battlefield"}:
            raise EnvConfigError(
                f"Invalid coordinate_mode: {self.coordinate_mode}. "
                "Available modes are: toy, battlefield."
            )

        if self.max_time <= 0:
            raise EnvConfigError("max_time must be positive.")

        if self.time_step <= 0:
            raise EnvConfigError("time_step must be positive.")

        if self.uav_speed <= 0:
            raise EnvConfigError("uav_speed must be positive.")

        if self.attack_payload_per_uav < 0:
            raise EnvConfigError("attack_payload_per_uav must be non-negative.")

        if not 0.0 <= self.boundary_margin_ratio <= 1.0:
            raise EnvConfigError(
                "boundary_margin_ratio should be in [0, 1]."
            )

        self.bounds.validate()
        self.random_event_probabilities.validate()

    def with_inferred_bounds(
        self,
        dataframes: list[pd.DataFrame],
    ) -> "EnvConfig":
        """
        根据真实数据返回一个边界自动扩展后的 EnvConfig。

        注意：这里返回新的 EnvConfig，而不是直接修改当前对象。
        这样可以避免外部代码在不知情的情况下改变配置状态，
        也方便后续在日志中记录“原始配置”和“实际使用配置”的差异。
        """
        if not self.auto_infer_bounds_from_data:
            return self

        inferred_bounds = self.bounds.expanded_by_dataframe(
            dataframes=dataframes,
            margin_ratio=self.boundary_margin_ratio,
        )

        new_config = EnvConfig(
            name=self.name,
            coordinate_mode=self.coordinate_mode,
            coordinate_unit=self.coordinate_unit,
            bounds=inferred_bounds,
            auto_infer_bounds_from_data=self.auto_infer_bounds_from_data,
            boundary_margin_ratio=self.boundary_margin_ratio,
            max_time=self.max_time,
            time_step=self.time_step,
            uav_speed=self.uav_speed,
            base_position=self.base_position,
            attack_payload_per_uav=self.attack_payload_per_uav,
            communication_covers_battlefield=self.communication_covers_battlefield,
            enable_dynamic_events=self.enable_dynamic_events,
            event_schedule=self.event_schedule,
            random_event_probabilities=self.random_event_probabilities,
        )
        new_config.validate()

        return new_config


def load_env_config(config: dict[str, Any]) -> EnvConfig:
    """
    从项目总配置中解析 env 配置。

    这里不让 DroneBattleEnv 直接依赖原始 dict，是为了让环境模块只面对结构化配置对象。
    这样后面不管配置文件如何拆分，环境模块的初始化接口都可以保持稳定。
    """
    schedule = DynamicEventSchedule(
        target_disappear_times=get_config_value(
            config,
            "env.event_schedule.target_disappear_times",
            default=[],
        ),
        target_appear_times=get_config_value(
            config,
            "env.event_schedule.target_appear_times",
            default=[],
        ),
        attack_uav_destroyed_times=get_config_value(
            config,
            "env.event_schedule.attack_uav_destroyed_times",
            default=[],
        ),
        guide_uav_destroyed_times=get_config_value(
            config,
            "env.event_schedule.guide_uav_destroyed_times",
            default=[],
        ),
    )

    random_probs = RandomEventProbabilities(
        target_appear=float(
            get_config_value(
                config,
                "env.random_event_probabilities.target_appear",
                default=0.0,
            )
        ),
        target_disappear=float(
            get_config_value(
                config,
                "env.random_event_probabilities.target_disappear",
                default=0.0,
            )
        ),
        uav_destroyed=float(
            get_config_value(
                config,
                "env.random_event_probabilities.uav_destroyed",
                default=0.0,
            )
        ),
    )

    env_config = EnvConfig(
        name=str(get_config_value(config, "env.name", default="drone_battle")),
        coordinate_mode=str(
            get_config_value(config, "env.coordinate_mode", default="battlefield")
        ),
        coordinate_unit=str(
            get_config_value(config, "env.coordinate_unit", default="m")
        ),
        bounds=BattlefieldBounds(
            x_min=float(get_config_value(config, "env.x_min", default=-2000.0)),
            x_max=float(get_config_value(config, "env.x_max", default=40000.0)),
            y_min=float(get_config_value(config, "env.y_min", default=-170000.0)),
            y_max=float(get_config_value(config, "env.y_max", default=2000.0)),
        ),
        auto_infer_bounds_from_data=bool(
            get_config_value(
                config,
                "env.auto_infer_bounds_from_data",
                default=True,
            )
        ),
        boundary_margin_ratio=float(
            get_config_value(
                config,
                "env.boundary_margin_ratio",
                default=0.05,
            )
        ),
        max_time=float(get_config_value(config, "env.max_time", default=2200.0)),
        time_step=float(get_config_value(config, "env.time_step", default=30.0)),
        uav_speed=float(get_config_value(config, "env.uav_speed", default=70.0)),
        base_position=Position(
            x=float(get_config_value(config, "env.base_x", default=0.0)),
            y=float(get_config_value(config, "env.base_y", default=-16000.0)),
        ),
        attack_payload_per_uav=float(
            get_config_value(
                config,
                "env.attack_payload_per_uav",
                default=6.0,
            )
        ),
        communication_covers_battlefield=bool(
            get_config_value(
                config,
                "env.communication_covers_battlefield",
                default=True,
            )
        ),
        enable_dynamic_events=bool(
            get_config_value(
                config,
                "env.enable_dynamic_events",
                default=False,
            )
        ),
        event_schedule=schedule,
        random_event_probabilities=random_probs,
    )

    env_config.validate()

    return env_config