"""envs 数据模块中的打击顺序环境实现。"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np

from uav_dynamic_task_allocation.core.contracts import TargetCluster
from uav_dynamic_task_allocation.core.entities import Position, Target
from uav_dynamic_task_allocation.utils.config import get_config_value


class StrikeOrderEnvError(Exception):
    """目标群内部打击次序环境相关的自定义错误。"""


@dataclass(frozen=True)
class StrikeOrderEnvConfig:
    """
    目标群内部打击次序环境配置。

    这个环境对应原代码 attack_order_dqn.py 的核心思想：
    给定一个目标群，智能体每一步选择下一个要访问 / 打击的目标，
    最终形成该目标群内部的打击顺序。

    注意：
    这里不模拟 UAV 从基地飞到战场的过程。
    在 DQN 训练阶段，我们假设 UAV 小组已经到达目标群任务区，
    或者已经进入可以执行群内打击排序的阶段。
    """

    # max_targets_per_cluster: 最大值目标集合per目标簇。
    max_targets_per_cluster: int = 50
    # include_distance_features: includedistancefeatures。
    include_distance_features: bool = True
    # distance_normalizer: distancenormalizer。
    distance_normalizer: str | float = "auto"

    # reward_mode: 奖励mode。
    reward_mode: str = "inverse_distance"
    # distance_reward_scale: distance奖励scale。
    distance_reward_scale: float = 100.0
    # repeat_penalty: repeatpenalty。
    repeat_penalty: float = -10.0
    # completion_bonus: completionbonus。
    completion_bonus: float = 10.0
    # priority_weight: priority权重。
    priority_weight: float = 0.1
    # target_value_normalizer: 目标数值normalizer。
    target_value_normalizer: float = 100.0

    def validate(self) -> None:
        """检查配置是否合法。"""
        if self.max_targets_per_cluster <= 0:
            raise StrikeOrderEnvError("max_targets_per_cluster must be positive.")

        if self.reward_mode not in {"inverse_distance", "negative_distance"}:
            raise StrikeOrderEnvError(
                f"Unsupported reward_mode: {self.reward_mode}. "
                "Available modes: inverse_distance, negative_distance."
            )

        if self.distance_reward_scale <= 0:
            raise StrikeOrderEnvError("distance_reward_scale must be positive.")

        if self.target_value_normalizer <= 0:
            raise StrikeOrderEnvError("target_value_normalizer must be positive.")


@dataclass(frozen=True)
class StrikeOrderObservation:
    """
    StrikeOrderEnv 的 observation。

    vector:
        固定长度状态向量，可直接输入 DQN 网络。

    action_mask:
        固定长度动作 mask。1 表示该目标当前可选，0 表示 padding 或已访问。

    feature_names:
        每一维特征名，调试时使用。

    debug_info:
        额外调试信息。
    """

    # vector: vector 数据。
    vector: np.ndarray
    # action_mask: 动作掩码。
    action_mask: np.ndarray
    # feature_names: featurenames。
    feature_names: list[str] = field(default_factory=list)
    # debug_info: 调试info。
    debug_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrikeOrderStepResult:
    """
    StrikeOrderEnv 单步执行结果。

    observation:
        动作执行后的 observation。

    reward:
        当前动作获得的 reward。

    done:
        当前目标群是否已经排序完成。

    info:
        调试信息，包括当前路径、距离、是否重复访问等。
    """

    # observation: 观测向量。
    observation: StrikeOrderObservation
    # reward: 奖励。
    reward: float
    # done: 结束标记。
    done: bool
    # info: info 数据。
    info: dict[str, Any]


class StrikeOrderEnv:
    """
    目标群内部打击次序决策环境。

    这个环境用于复现和优化原 attack_order_dqn.py 的核心任务：
    给定一个目标群，DQN 每一步选择一个未访问目标，
    形成目标群内部的打击顺序。

    与之前的 DroneBattleEnv 不同：
    - DroneBattleEnv 的 action 是 UAV-Target；
    - StrikeOrderEnv 的 action 是 target_slot；
    - DroneBattleEnv 面向全局 UAV-Target baseline；
    - StrikeOrderEnv 面向源代码中的群内打击排序主线。

    训练时可以理解为：
    UAV 小组已经到达目标群附近，现在要决定先打哪个、后打哪个。
    """

    def __init__(
        self,
        target_cluster: TargetCluster,
        start_position: Position,
        config: StrikeOrderEnvConfig,
    ) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            target_cluster: target_cluster 参数，类型为 TargetCluster。
            start_position: start_position 参数，类型为 Position。
            config: 配置对象，类型为 StrikeOrderEnvConfig。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        config.validate()

        # target_cluster: 目标目标簇。
        self.target_cluster = target_cluster
        # start_position: start位置坐标。
        self.start_position = start_position
        # config: 配置。
        self.config = config

        if len(self.target_cluster.targets) > self.config.max_targets_per_cluster:
            raise StrikeOrderEnvError(
                "Number of targets in cluster exceeds max_targets_per_cluster: "
                f"num_targets={len(self.target_cluster.targets)}, "
                f"max={self.config.max_targets_per_cluster}"
            )

        # distance_normalizer: distancenormalizer。
        self.distance_normalizer = self._resolve_distance_normalizer()

        # current_position: 当前位置坐标。
        self.current_position: Position = start_position
        # visited: visited 数据。
        self.visited: np.ndarray = np.zeros(
            self.config.max_targets_per_cluster,
            dtype=np.float32,
        )
        # path_target_ids: 路径目标编号集合。
        self.path_target_ids: list[int] = []
        # path_positions: 路径positions。
        self.path_positions: list[Position] = []
        # total_path_distance: total路径distance。
        self.total_path_distance: float = 0.0
        # current_step: 当前步数。
        self.current_step: int = 0
        # episode_reward: 训练回合奖励。
        self.episode_reward: float = 0.0

    @property
    def action_dim(self) -> int:
        """
        返回动作维度。

        对 StrikeOrderEnv 来说，DQN 输出层维度应该等于 max_targets_per_cluster。
        action_id = target_slot。
        """
        return self.config.max_targets_per_cluster

    @property
    def num_real_targets(self) -> int:
        """返回当前目标群中的真实目标数量。"""
        return len(self.target_cluster.targets)

    def reset(self) -> StrikeOrderObservation:
        """
        重置目标群排序环境。

        每个 episode 开始时：
        - 当前位置回到 start_position；
        - 所有目标设为未访问；
        - 清空路径；
        - 清空累计距离和 reward。
        """
        self.current_position = self.start_position
        self.visited = np.zeros(
            self.config.max_targets_per_cluster,
            dtype=np.float32,
        )
        self.path_target_ids = []
        self.path_positions = [self.start_position]
        self.total_path_distance = 0.0
        self.current_step = 0
        self.episode_reward = 0.0

        return self._build_observation()

    def step(self, action_id: int) -> StrikeOrderStepResult:
        """
        执行一个群内目标选择动作。

        Args:
            action_id:
                目标在当前 cluster 中的 slot 编号。
                不是 target_id，而是目标群内部的 target index。

        Returns:
            StrikeOrderStepResult。
        """
        self.current_step += 1

        info: dict[str, Any] = {
            "cluster_id": self.target_cluster.cluster_id,
            "action_id": action_id,
            "event": None,
            "message": "",
            "current_step": self.current_step,
        }

        if action_id < 0 or action_id >= self.config.max_targets_per_cluster:
            reward = self.config.repeat_penalty
            info["event"] = "invalid_action_id"
            info["message"] = f"Invalid action_id: {action_id}."

            self.episode_reward += reward
            return self._make_step_result(reward=reward, info=info)

        if action_id >= self.num_real_targets:
            reward = self.config.repeat_penalty
            info["event"] = "padding_target_slot"
            info["message"] = f"Action {action_id} points to a padding target slot."

            self.episode_reward += reward
            return self._make_step_result(reward=reward, info=info)

        if self.visited[action_id] > 0:
            reward = self.config.repeat_penalty
            target = self.target_cluster.targets[action_id]

            info["event"] = "repeated_target"
            info["message"] = (
                f"Target slot {action_id}, target_id={target.target_id} "
                "has already been visited."
            )
            info["target_id"] = target.target_id

            self.episode_reward += reward
            return self._make_step_result(reward=reward, info=info)

        target = self.target_cluster.targets[action_id]
        distance = self._distance(self.current_position, target.position)

        reward = self._calculate_valid_action_reward(
            target=target,
            distance=distance,
        )

        self.visited[action_id] = 1.0
        self.current_position = target.position
        self.path_target_ids.append(target.target_id)
        self.path_positions.append(target.position)
        self.total_path_distance += distance

        done = self._is_done()

        if done:
            reward += self.config.completion_bonus
            info["completion_bonus"] = self.config.completion_bonus

        info.update(
            {
                "event": "target_selected",
                "message": (
                    f"Selected target_id={target.target_id} "
                    f"at slot={action_id}."
                ),
                "target_id": target.target_id,
                "distance": distance,
                "total_path_distance": self.total_path_distance,
                "path_target_ids": list(self.path_target_ids),
                "num_visited_targets": int(self.visited[: self.num_real_targets].sum()),
                "done": done,
            }
        )

        self.episode_reward += reward
        return self._make_step_result(reward=reward, info=info)

    def get_action_mask(self) -> np.ndarray:
        """
        返回当前动作 mask。

        1 表示该 target slot 是真实目标且尚未访问；
        0 表示该 slot 是 padding 或已经访问。
        """
        mask = np.zeros(
            self.config.max_targets_per_cluster,
            dtype=np.float32,
        )

        for index in range(self.num_real_targets):
            if self.visited[index] <= 0:
                mask[index] = 1.0

        return mask

    def get_strike_order_plan_dict(self) -> dict[str, Any]:
        """
        返回当前路径规划结果的字典形式。

        后面 planning/strike_order_planner.py 会把它进一步包装成 StrikeOrderPlan。
        """
        return {
            "cluster_id": self.target_cluster.cluster_id,
            "ordered_target_ids": list(self.path_target_ids),
            "total_path_distance": self.total_path_distance,
            "episode_reward": self.episode_reward,
            "num_targets": self.num_real_targets,
            "num_visited_targets": int(self.visited[: self.num_real_targets].sum()),
        }

    def _build_observation(self) -> StrikeOrderObservation:
        """
        构造固定长度 observation。

        当前 observation 包括：
        1. 当前 UAV / 集群位置；
        2. 每个目标的位置、重要性、防御值、访问状态；
        3. 可选：当前位置到每个目标的距离；
        4. 全局进度特征。

        这个 observation 比原 attack_order_dqn.py 更工程化：
        原代码主要把目标坐标和当前位置拼接；
        这里额外加入 visited mask、目标价值、防御值和进度信息。
        """
        features: list[float] = []
        feature_names: list[str] = []

        # 全局当前位置和进度特征。
        features.extend(
            [
                self._normalize_x(self.current_position.x),
                self._normalize_y(self.current_position.y),
                self._safe_divide(
                    self.visited[: self.num_real_targets].sum(),
                    max(self.num_real_targets, 1),
                ),
                self._safe_divide(
                    self.current_step,
                    max(self.num_real_targets, 1),
                ),
            ]
        )
        feature_names.extend(
            [
                "global.current_x_norm",
                "global.current_y_norm",
                "global.visited_ratio",
                "global.step_ratio",
            ]
        )

        for index in range(self.config.max_targets_per_cluster):
            if index < self.num_real_targets:
                target = self.target_cluster.targets[index]

                distance = self._distance(
                    self.current_position,
                    target.position,
                )

                target_features = [
                    self._normalize_x(target.position.x),
                    self._normalize_y(target.position.y),
                    self._normalize_positive(float(target.target_type)),
                    self._normalize_positive(target.defense),
                    self._normalize_positive(target.significance),
                    float(self.visited[index]),
                ]

                if self.config.include_distance_features:
                    target_features.append(
                        self._safe_divide(distance, self.distance_normalizer)
                    )
            else:
                target_features = [0.0] * 6
                if self.config.include_distance_features:
                    target_features.append(0.0)

            features.extend(target_features)

            base_names = [
                f"target[{index}].x_norm",
                f"target[{index}].y_norm",
                f"target[{index}].target_type_norm",
                f"target[{index}].defense_norm",
                f"target[{index}].significance_norm",
                f"target[{index}].visited",
            ]

            if self.config.include_distance_features:
                base_names.append(f"target[{index}].distance_to_current_norm")

            feature_names.extend(base_names)

        vector = np.array(features, dtype=np.float32)
        action_mask = self.get_action_mask()

        debug_info = {
            "cluster_id": self.target_cluster.cluster_id,
            "input_dim": int(vector.shape[0]),
            "action_dim": int(action_mask.shape[0]),
            "num_real_targets": self.num_real_targets,
            "num_visited_targets": int(self.visited[: self.num_real_targets].sum()),
            "current_step": self.current_step,
            "episode_reward": self.episode_reward,
            "total_path_distance": self.total_path_distance,
            "distance_normalizer": self.distance_normalizer,
        }

        return StrikeOrderObservation(
            vector=vector,
            action_mask=action_mask,
            feature_names=feature_names,
            debug_info=debug_info,
        )

    def _calculate_valid_action_reward(
        self,
        target: Target,
        distance: float,
    ) -> float:
        """
        计算选择一个未访问目标的 reward。

        inverse_distance 模式更接近原代码：
            reward = 100 / distance

        negative_distance 模式更接近现代 RL 工程：
            reward = - normalized_distance + priority_reward

        当前默认 inverse_distance，是为了先和源代码 attack_order_dqn.py 对齐。
        同时加入较小的 priority_reward，让目标重要性参与排序。
        """
        if self.config.reward_mode == "inverse_distance":
            distance_reward = self.config.distance_reward_scale / max(distance, 1e-5)
        elif self.config.reward_mode == "negative_distance":
            distance_reward = -self._safe_divide(distance, self.distance_normalizer)
        else:
            raise StrikeOrderEnvError(
                f"Unsupported reward_mode: {self.config.reward_mode}"
            )

        target_value = getattr(target, "value_score", target.significance)

        priority_reward = (
            self.config.priority_weight
            * self._safe_divide(
                target_value,
                self.config.target_value_normalizer,
            )
        )

        return float(distance_reward + priority_reward)

    def _make_step_result(
        self,
        reward: float,
        info: dict[str, Any],
    ) -> StrikeOrderStepResult:
        """统一构造 step 返回结果。"""
        done = self._is_done()
        info["done"] = done
        info["episode_reward"] = self.episode_reward

        return StrikeOrderStepResult(
            observation=self._build_observation(),
            reward=float(reward),
            done=done,
            info=info,
        )

    def _is_done(self) -> bool:
        """判断当前目标群是否已经全部访问完成。"""
        if self.num_real_targets == 0:
            return True

        return bool(
            self.visited[: self.num_real_targets].sum() >= self.num_real_targets
        )

    def _resolve_distance_normalizer(self) -> float:
        """
        解析距离归一化尺度。

        auto 模式下，根据目标群坐标范围估计距离尺度。
        如果目标群范围非常小，则回退到 1.0，避免除零。
        """
        if self.config.distance_normalizer != "auto":
            value = float(self.config.distance_normalizer)
            if value <= 0:
                raise StrikeOrderEnvError("distance_normalizer must be positive.")
            return value

        positions = [target.position for target in self.target_cluster.targets]
        positions.append(self.start_position)

        x_values = [position.x for position in positions]
        y_values = [position.y for position in positions]

        x_span = max(x_values) - min(x_values)
        y_span = max(y_values) - min(y_values)

        diagonal = sqrt(x_span**2 + y_span**2)

        return max(diagonal, 1.0)

    def _normalize_x(self, x: float) -> float:
        """根据目标群范围对 x 坐标做局部归一化。"""
        x_values = [target.position.x for target in self.target_cluster.targets]
        x_values.append(self.start_position.x)

        x_min = min(x_values)
        x_max = max(x_values)

        return self._safe_divide(x - x_min, max(x_max - x_min, 1e-8))

    def _normalize_y(self, y: float) -> float:
        """根据目标群范围对 y 坐标做局部归一化。"""
        y_values = [target.position.y for target in self.target_cluster.targets]
        y_values.append(self.start_position.y)

        y_min = min(y_values)
        y_max = max(y_values)

        return self._safe_divide(y - y_min, max(y_max - y_min, 1e-8))

    @staticmethod
    def _normalize_positive(value: float, scale: float = 100.0) -> float:
        """对正值特征做简单归一化。"""
        return float(value) / scale

    @staticmethod
    def _distance(position_a: Position, position_b: Position) -> float:
        """计算两个位置之间的欧氏距离。"""
        return sqrt(
            (position_a.x - position_b.x) ** 2
            + (position_a.y - position_b.y) ** 2
        )

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """安全除法。"""
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)


def load_strike_order_env_config(config: dict[str, Any]) -> StrikeOrderEnvConfig:
    """
    从项目总配置中读取 StrikeOrderEnv 配置。
    """
    reward_prefix = "strike_order.reward"

    env_config = StrikeOrderEnvConfig(
        max_targets_per_cluster=int(
            get_config_value(
                config,
                "strike_order.max_targets_per_cluster",
                default=50,
            )
        ),
        include_distance_features=bool(
            get_config_value(
                config,
                "strike_order.include_distance_features",
                default=True,
            )
        ),
        distance_normalizer=get_config_value(
            config,
            "strike_order.distance_normalizer",
            default="auto",
        ),
        reward_mode=str(
            get_config_value(
                config,
                f"{reward_prefix}.mode",
                default="inverse_distance",
            )
        ),
        distance_reward_scale=float(
            get_config_value(
                config,
                f"{reward_prefix}.distance_reward_scale",
                default=100.0,
            )
        ),
        repeat_penalty=float(
            get_config_value(
                config,
                f"{reward_prefix}.repeat_penalty",
                default=-10.0,
            )
        ),
        completion_bonus=float(
            get_config_value(
                config,
                f"{reward_prefix}.completion_bonus",
                default=10.0,
            )
        ),
        priority_weight=float(
            get_config_value(
                config,
                f"{reward_prefix}.priority_weight",
                default=0.1,
            )
        ),
        target_value_normalizer=float(
            get_config_value(
                config,
                f"{reward_prefix}.target_value_normalizer",
                default=100.0,
            )
        ),
    )

    env_config.validate()
    return env_config