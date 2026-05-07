from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

from uav_dynamic_task_allocation.core.entities import (
    BattlefieldState,
    EntityStatus,
    Target,
    UAV,
    UAVType,
)
from uav_dynamic_task_allocation.envs.env_config import EnvConfig
from uav_dynamic_task_allocation.utils.config import get_config_value


class ObservationError(Exception):
    """observation 构造过程中的自定义错误。"""


@dataclass(frozen=True)
class ObservationConfig:
    """
    observation 构造配置。

    这个配置对象用于控制状态向量的维度和特征组成。
    强化学习模型要求输入维度固定，而动态战场中的 UAV 和 Target 数量会变化，
    因此这里必须显式规定 max_uavs 和 max_targets。
    """

    max_uavs: int = 50
    max_targets: int = 120
    include_distance_matrix: bool = True
    include_feature_names: bool = True
    distance_normalizer: str | float = "auto"

    def validate(self) -> None:
        """检查 observation 配置是否合法。"""
        if self.max_uavs <= 0:
            raise ObservationError("max_uavs must be positive.")

        if self.max_targets <= 0:
            raise ObservationError("max_targets must be positive.")


@dataclass(frozen=True)
class Observation:
    """
    模型可用的 observation 结果。

    vector:
        固定长度数值向量，可直接作为 DQN/PPO/SAC 的输入。
    feature_names:
        每个向量维度对应的含义，主要用于调试和解释模型输入。
    masks:
        UAV 和 Target 的有效位置标记。1 表示真实对象，0 表示 padding。
    debug_info:
        额外调试信息，例如输入维度、截断数量、归一化参数等。
    """

    vector: np.ndarray
    feature_names: list[str]
    masks: dict[str, np.ndarray]
    debug_info: dict[str, Any]


class ObservationBuilder:
    """
    战场状态 observation 构造器。

    该类负责将 BattlefieldState 转换成固定长度数值向量。
    它是环境模块和强化学习算法之间的接口层：

        BattlefieldState
              ↓
        ObservationBuilder
              ↓
        np.ndarray

    为什么不直接在 DroneBattleEnv._get_observation() 里写？
    因为 observation 特征会不断扩展，例如目标分群特征、资源评估特征、
    距离矩阵、动作 mask、注意力输入等。如果全部写在环境类里，
    DroneBattleEnv 会越来越臃肿，不利于后续维护。
    """

    def __init__(
        self,
        env_config: EnvConfig,
        obs_config: ObservationConfig,
    ) -> None:
        self.env_config = env_config
        self.obs_config = obs_config
        self.obs_config.validate()

        self.distance_normalizer = self._resolve_distance_normalizer()

    def build(self, state: BattlefieldState) -> Observation:
        """
        根据当前战场状态构造固定长度 observation。

        当前 observation 由四部分组成：
        1. 全局特征 global features；
        2. UAV 特征序列；
        3. Target 特征序列；
        4. UAV-Target 距离矩阵特征，可选。

        这些特征对应论文中的态势建模思想：
        目标位置、防御能力、重要程度、动态状态，以及无人机资源状态。
        """
        global_vector, global_names = self._build_global_features(state)
        uav_vector, uav_names, uav_mask = self._build_uav_features(state)
        target_vector, target_names, target_mask = self._build_target_features(state)

        vectors = [global_vector, uav_vector, target_vector]
        feature_names = global_names + uav_names + target_names

        masks = {
            "uav_mask": uav_mask,
            "target_mask": target_mask,
        }

        if self.obs_config.include_distance_matrix:
            distance_vector, distance_names = self._build_distance_matrix_features(
                state=state,
            )
            vectors.append(distance_vector)
            feature_names.extend(distance_names)

        vector = np.concatenate(vectors).astype(np.float32)

        debug_info = {
            "input_dim": int(vector.shape[0]),
            "num_uavs": len(state.uavs),
            "num_targets": len(state.targets),
            "max_uavs": self.obs_config.max_uavs,
            "max_targets": self.obs_config.max_targets,
            "num_active_uavs": len(state.active_uavs),
            "num_active_targets": len(state.active_targets),
            "distance_normalizer": self.distance_normalizer,
            "include_distance_matrix": self.obs_config.include_distance_matrix,
        }

        return Observation(
            vector=vector,
            feature_names=feature_names
            if self.obs_config.include_feature_names
            else [],
            masks=masks,
            debug_info=debug_info,
        )

    def get_input_dim(self) -> int:
        """
        返回当前 observation 配置下的输入维度。

        训练 DQN 时，这个值会作为神经网络输入层维度。
        这里通过构造特征维度公式计算，而不是依赖某个具体 state。
        """
        global_dim = 6

        # UAV 特征：
        # x, y, type_guide, type_comm, type_attack, work_range,
        # attack_power, active, destroyed
        uav_feature_dim = 9

        # Target 特征：
        # x, y, target_type, defense, significance,
        # active, destroyed, inactive
        target_feature_dim = 8

        input_dim = (
            global_dim
            + self.obs_config.max_uavs * uav_feature_dim
            + self.obs_config.max_targets * target_feature_dim
        )

        if self.obs_config.include_distance_matrix:
            input_dim += self.obs_config.max_uavs * self.obs_config.max_targets

        return input_dim

    def _build_global_features(
        self,
        state: BattlefieldState,
    ) -> tuple[np.ndarray, list[str]]:
        """
        构造全局战场特征。

        全局特征用于告诉模型当前 episode 处于什么阶段，以及整体资源是否充足。
        注意这里不放 episode_reward，因为 reward 是训练反馈，不是环境状态本身。
        """
        current_time = state.current_step * self.env_config.time_step

        features = np.array(
            [
                self._safe_divide(current_time, self.env_config.max_time),
                self._safe_divide(state.current_step, self.env_config.max_steps),
                self._safe_divide(
                    len(state.active_uavs),
                    max(len(state.uavs), 1),
                ),
                self._safe_divide(
                    len(state.attack_uavs),
                    max(len(state.uavs), 1),
                ),
                self._safe_divide(
                    len(state.active_targets),
                    max(len(state.targets), 1),
                ),
                1.0 if self.env_config.enable_dynamic_events else 0.0,
            ],
            dtype=np.float32,
        )

        names = [
            "global.current_time_ratio",
            "global.current_step_ratio",
            "global.active_uav_ratio",
            "global.attack_uav_ratio",
            "global.active_target_ratio",
            "global.dynamic_events_enabled",
        ]

        return features, names

    def _build_uav_features(
        self,
        state: BattlefieldState,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """
        构造 UAV 特征序列。

        每架 UAV 使用 9 个特征：
        1. 归一化 x；
        2. 归一化 y；
        3. 是否导引 UAV；
        4. 是否通信 UAV；
        5. 是否攻击 UAV；
        6. 工作距离归一化；
        7. 攻击能力归一化；
        8. 是否 active；
        9. 是否 destroyed。

        如果 UAV 数量小于 max_uavs，后面用 0 padding。
        如果超过 max_uavs，当前先截断，后续可以通过配置增大 max_uavs。
        """
        feature_blocks: list[list[float]] = []
        feature_names: list[str] = []
        mask = np.zeros(self.obs_config.max_uavs, dtype=np.float32)

        selected_uavs = state.uavs[: self.obs_config.max_uavs]

        for index in range(self.obs_config.max_uavs):
            if index < len(selected_uavs):
                uav = selected_uavs[index]
                mask[index] = 1.0
                block = self._encode_uav(uav)
            else:
                block = [0.0] * 9

            feature_blocks.append(block)
            feature_names.extend(
                [
                    f"uav[{index}].x_norm",
                    f"uav[{index}].y_norm",
                    f"uav[{index}].type_guide",
                    f"uav[{index}].type_communication",
                    f"uav[{index}].type_attack",
                    f"uav[{index}].work_range_norm",
                    f"uav[{index}].attack_power_norm",
                    f"uav[{index}].is_active",
                    f"uav[{index}].is_destroyed",
                ]
            )

        return (
            np.array(feature_blocks, dtype=np.float32).reshape(-1),
            feature_names,
            mask,
        )

    def _build_target_features(
        self,
        state: BattlefieldState,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """
        构造 Target 特征序列。

        每个目标使用 8 个特征：
        1. 归一化 x；
        2. 归一化 y；
        3. 目标类型归一化；
        4. 防御能力归一化；
        5. 重要程度归一化；
        6. 是否 active；
        7. 是否 destroyed；
        8. 是否 inactive。

        这些特征对应论文中目标态势模型的核心变量：
        位置、类型、防御能力、战略价值和动态状态。
        """
        feature_blocks: list[list[float]] = []
        feature_names: list[str] = []
        mask = np.zeros(self.obs_config.max_targets, dtype=np.float32)

        selected_targets = state.targets[: self.obs_config.max_targets]

        for index in range(self.obs_config.max_targets):
            if index < len(selected_targets):
                target = selected_targets[index]
                mask[index] = 1.0
                block = self._encode_target(target)
            else:
                block = [0.0] * 8

            feature_blocks.append(block)
            feature_names.extend(
                [
                    f"target[{index}].x_norm",
                    f"target[{index}].y_norm",
                    f"target[{index}].target_type_norm",
                    f"target[{index}].defense_norm",
                    f"target[{index}].significance_norm",
                    f"target[{index}].is_active",
                    f"target[{index}].is_destroyed",
                    f"target[{index}].is_inactive",
                ]
            )

        return (
            np.array(feature_blocks, dtype=np.float32).reshape(-1),
            feature_names,
            mask,
        )

    def _build_distance_matrix_features(
        self,
        state: BattlefieldState,
    ) -> tuple[np.ndarray, list[str]]:
        """
        构造 UAV-Target 距离矩阵特征。

        距离是打击次序决策中的关键因素。论文奖励函数中包含路径距离项，
        因此模型输入中保留 UAV 与 Target 的归一化距离，有利于后续 DQN
        学习“高价值目标”和“路径代价”之间的权衡关系。

        输出维度为：
            max_uavs * max_targets
        """
        matrix = np.zeros(
            (self.obs_config.max_uavs, self.obs_config.max_targets),
            dtype=np.float32,
        )
        feature_names: list[str] = []

        selected_uavs = state.uavs[: self.obs_config.max_uavs]
        selected_targets = state.targets[: self.obs_config.max_targets]

        for uav_index in range(self.obs_config.max_uavs):
            for target_index in range(self.obs_config.max_targets):
                feature_names.append(
                    f"distance.uav[{uav_index}].target[{target_index}]"
                )

                if (
                    uav_index >= len(selected_uavs)
                    or target_index >= len(selected_targets)
                ):
                    continue

                uav = selected_uavs[uav_index]
                target = selected_targets[target_index]

                distance = uav.distance_to_target(target)
                matrix[uav_index, target_index] = self._safe_divide(
                    distance,
                    self.distance_normalizer,
                )

        return matrix.reshape(-1), feature_names

    def _encode_uav(self, uav: UAV) -> list[float]:
        """
        将单架 UAV 编码为固定长度数值特征。

        UAV 类型使用 one-hot，而不是直接使用 1/2/3。
        这样做可以避免模型误以为 ATTACK=3 比 GUIDE=1 “数值更大更重要”。
        """
        return [
            self._normalize_x(uav.position.x),
            self._normalize_y(uav.position.y),
            1.0 if uav.uav_type == UAVType.GUIDE else 0.0,
            1.0 if uav.uav_type == UAVType.COMMUNICATION else 0.0,
            1.0 if uav.uav_type == UAVType.ATTACK else 0.0,
            self._safe_divide(uav.work_range, self.distance_normalizer),
            self._normalize_positive(uav.attack_power),
            1.0 if uav.status == EntityStatus.ACTIVE else 0.0,
            1.0 if uav.status == EntityStatus.DESTROYED else 0.0,
        ]

    def _encode_target(self, target: Target) -> list[float]:
        """
        将单个目标编码为固定长度数值特征。

        这里暂时对 defense 和 significance 使用简单正值归一化。
        后续如果确定论文或数据中有更严格的最大值定义，可以替换成更精确的归一化策略。
        """
        return [
            self._normalize_x(target.position.x),
            self._normalize_y(target.position.y),
            self._normalize_positive(float(target.target_type)),
            self._normalize_positive(target.defense),
            self._normalize_positive(target.significance),
            1.0 if target.status == EntityStatus.ACTIVE else 0.0,
            1.0 if target.status == EntityStatus.DESTROYED else 0.0,
            1.0 if target.status == EntityStatus.INACTIVE else 0.0,
        ]

    def _normalize_x(self, x: float) -> float:
        """将 x 坐标归一化到大致 0~1 区间。"""
        return self._safe_divide(
            x - self.env_config.bounds.x_min,
            self.env_config.bounds.x_max - self.env_config.bounds.x_min,
        )

    def _normalize_y(self, y: float) -> float:
        """将 y 坐标归一化到大致 0~1 区间。"""
        return self._safe_divide(
            y - self.env_config.bounds.y_min,
            self.env_config.bounds.y_max - self.env_config.bounds.y_min,
        )

    @staticmethod
    def _normalize_positive(value: float, scale: float = 100.0) -> float:
        """
        对正值特征做简单归一化。

        当前数据中 target defense、significance、attack_power 等多为正值，
        暂时用 100 作为基础尺度。后续可以根据训练集统计量改成 max-normalization
        或 standardization。
        """
        return float(value) / scale

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """安全除法，避免 denominator 为 0 导致运行崩溃。"""
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)

    def _resolve_distance_normalizer(self) -> float:
        """
        解析距离归一化尺度。

        如果配置为 auto，则使用战场边界对角线作为最大距离尺度。
        这样无论 toy 坐标还是真实战场坐标，都能得到相对稳定的距离特征。
        """
        if self.obs_config.distance_normalizer == "auto":
            x_span = self.env_config.bounds.x_max - self.env_config.bounds.x_min
            y_span = self.env_config.bounds.y_max - self.env_config.bounds.y_min
            return sqrt(x_span**2 + y_span**2)

        value = float(self.obs_config.distance_normalizer)
        if value <= 0:
            raise ObservationError("distance_normalizer must be positive.")

        return value


def load_observation_config(config: dict[str, Any]) -> ObservationConfig:
    """
    从项目总配置中读取 observation 配置。

    这样做可以让 observation 的维度、是否包含距离矩阵、是否输出特征名
    都由 YAML 控制，而不是写死在代码里。
    """
    obs_config = ObservationConfig(
        max_uavs=int(get_config_value(config, "observation.max_uavs", default=50)),
        max_targets=int(
            get_config_value(config, "observation.max_targets", default=120)
        ),
        include_distance_matrix=bool(
            get_config_value(
                config,
                "observation.include_distance_matrix",
                default=True,
            )
        ),
        include_feature_names=bool(
            get_config_value(
                config,
                "observation.include_feature_names",
                default=True,
            )
        ),
        distance_normalizer=get_config_value(
            config,
            "observation.distance_normalizer",
            default="auto",
        ),
    )
    obs_config.validate()

    return obs_config