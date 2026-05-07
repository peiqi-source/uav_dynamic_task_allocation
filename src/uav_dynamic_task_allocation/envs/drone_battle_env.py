from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pandas as pd

from uav_dynamic_task_allocation.core.entities import (
    BattlefieldState,
    EntityError,
    EntityStatus,
    Target,
    UAV,
    UAVType,
    build_battlefield_state,
)
from uav_dynamic_task_allocation.envs.env_config import EnvConfig
from uav_dynamic_task_allocation.envs.reward import (
    RewardBreakdown,
    RewardCalculator,
    RewardConfig,
    RewardContext,
)


class EnvironmentError(Exception):
    """战场环境运行过程中的自定义错误。"""


@dataclass
class AttackExecutionResult:
    """
    单次打击动作的执行结果。

    这个对象只描述“动作执行后发生了什么”，不直接计算 reward。
    这样设计是为了将环境状态更新和奖励函数计算分开：

    - 环境负责执行动作、修改 Target / UAV 状态；
    - RewardCalculator 负责根据执行结果计算奖励。

    这种解耦方式可以让后续 reward 设计持续调整，而不需要频繁改动环境主逻辑。
    """

    is_valid_action: bool
    invalid_reason: str | None = None

    target_was_active_before_action: bool = True
    target_destroyed_after_action: bool = False
    target_damaged_after_action: bool = False

    previous_target_defense: float | None = None
    remaining_target_defense: float | None = None


@dataclass
class StepResult:
    """
    环境单步执行结果。

    observation:
        当前动作执行后的环境观测。
    reward:
        当前动作的总奖励。这个值来自 RewardBreakdown.total_reward。
    done:
        当前 episode 是否结束。
    info:
        附加信息，包括动作是否合法、事件类型、文字说明、reward_breakdown 等。

    说明：
    强化学习训练时通常只直接使用 observation、reward、done；
    但在科研调试和论文实验分析中，info 非常重要，因为它能解释 reward 为什么变大或变小。
    """

    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class DroneBattleEnv:
    """
    UAV 集群对地目标任务决策环境。

    当前环境对应论文中“打击次序决策环境”的工程化基础版本。
    它负责维护 BattlefieldState，并通过 reset() / step() 接口与强化学习算法交互。

    当前版本的职责边界：
    1. 环境负责状态初始化、动作执行、状态推进和 episode 终止判断；
    2. RewardCalculator 负责奖励函数计算；
    3. ActionSpace 负责动作枚举与合法动作筛选；
    4. ObservationBuilder 负责将 BattlefieldState 转换为模型输入向量；
    5. 动态事件后续由 events/ 模块统一处理。

    这样拆分的原因是：原始代码中很多逻辑混在 main.py 和不同实验脚本里，
    后续很难维护。新工程要把“状态、动作、奖励、事件、算法”分层管理。
    """

    def __init__(
        self,
        uav_df: pd.DataFrame,
        target_df: pd.DataFrame,
        env_config: EnvConfig,
        reward_config: RewardConfig | None = None,
    ) -> None:
        """
        初始化战场环境。

        Args:
            uav_df:
                已经经过 loaders.py 标准化和 validators.py 校验的 UAV 数据。
            target_df:
                已经经过 loaders.py 标准化和 validators.py 校验的 Target 数据。
            env_config:
                环境配置对象，包含战场边界、时间步长、最大仿真时间、基地位置等参数。
            reward_config:
                奖励函数配置。如果不传入，则使用 RewardConfig 默认值。

        注意：
        这里不直接接收原始 config dict，而是接收结构化的 EnvConfig / RewardConfig。
        这是为了让环境类不依赖 YAML 的具体字段路径，提高模块稳定性。
        """
        self.uav_df = uav_df.copy()
        self.target_df = target_df.copy()

        # 根据真实 UAV / Target 数据自动扩展边界，避免手动配置边界过窄。
        self.env_config = env_config.with_inferred_bounds(
            dataframes=[self.uav_df, self.target_df]
        )

        self.reward_config = reward_config or RewardConfig()
        self.reward_calculator = RewardCalculator(
            env_config=self.env_config,
            reward_config=self.reward_config,
        )

        self.state: BattlefieldState | None = None

        # current_time 对应原始 main.py 中的 t。
        # current_step 是强化学习环境里更常用的离散步数。
        self.current_time: float = 0.0
        self.episode_reward: float = 0.0

        # 记录已经被选择打击过的目标编号。
        # 这对应论文 reward 中的重复打击惩罚 R_rep。
        # 注意：这里记录的是“已经尝试打击过”，不是“已经摧毁”。
        self.attacked_target_ids: set[int] = set()

    def reset(self) -> dict[str, Any]:
        """
        重置环境，生成新的初始战场状态。

        每个 episode 开始前调用 reset()。
        当前环境使用真实 CSV 数据构建初始状态，不随机生成 UAV 和 Target，
        这样可以保证工程重构不脱离原始代码和论文仿真数据。
        """
        self.state = build_battlefield_state(
            uav_df=self.uav_df,
            target_df=self.target_df,
        )

        self.current_time = 0.0
        self.episode_reward = 0.0
        self.attacked_target_ids = set()

        return self._get_observation()

    def step(self, action: dict[str, int]) -> StepResult:
        """
        执行一个 UAV-Target 动作，并推进环境一个时间步。

        当前动作格式：
            {
                "uav_id": 3,
                "target_id": 8
            }

        这个格式适合调试和日志记录。后续 DQN 输出 action_id 后，
        会通过 action_space.py 转换成这个可解释动作。
        """
        self._ensure_initialized()
        assert self.state is not None

        self.state.current_step += 1
        self.current_time = self.state.current_step * self.env_config.time_step

        info: dict[str, Any] = {
            "valid_action": True,
            "event": None,
            "message": "",
            "current_time": self.current_time,
            "current_step": self.state.current_step,
        }

        attacked_before = set(self.attacked_target_ids)

        try:
            uav = self.state.get_uav_by_id(action["uav_id"])
            target = self.state.get_target_by_id(action["target_id"])
        except KeyError as exc:
            info["valid_action"] = False
            info["event"] = "invalid_action_format"
            info["message"] = f"Action missing key: {exc}"

            reward_breakdown = self._calculate_reward_for_invalid_action(
                invalid_reason="invalid_action_format",
                attacked_before=attacked_before,
            )
            return self._build_step_result(reward_breakdown, info)

        except EntityError as exc:
            info["valid_action"] = False
            info["event"] = "entity_not_found"
            info["message"] = str(exc)

            reward_breakdown = self._calculate_reward_for_invalid_action(
                invalid_reason="entity_not_found",
                attacked_before=attacked_before,
            )
            return self._build_step_result(reward_breakdown, info)

        execution_result = self._execute_attack(
            uav=uav,
            target=target,
            info=info,
        )

        if execution_result.is_valid_action:
            # 只要是一次合法打击尝试，就将该目标加入 attacked history。
            # 这样后续再次选择同一目标时，会触发重复打击惩罚。
            self.attacked_target_ids.add(target.target_id)
        else:
            info["valid_action"] = False

        reward_context = RewardContext(
            uav=uav,
            target=target,
            current_step=self.state.current_step,
            current_time=self.current_time,
            is_valid_action=execution_result.is_valid_action,
            invalid_reason=execution_result.invalid_reason,
            target_was_active_before_action=(
                execution_result.target_was_active_before_action
            ),
            target_destroyed_after_action=(
                execution_result.target_destroyed_after_action
            ),
            target_damaged_after_action=(
                execution_result.target_damaged_after_action
            ),
            previous_target_defense=execution_result.previous_target_defense,
            remaining_target_defense=execution_result.remaining_target_defense,
            attacked_target_ids_before_action=attacked_before,
        )

        reward_breakdown = self.reward_calculator.calculate(reward_context)

        # 当前只保留动态事件配置入口，正式动态事件处理后续交给 events/ 模块。
        # 这样做是为了避免 DroneBattleEnv 变成一个过度臃肿的大类。
        info["dynamic_event_pending"] = bool(self.env_config.enable_dynamic_events)

        return self._build_step_result(reward_breakdown, info)

    def _execute_attack(
        self,
        uav: UAV,
        target: Target,
        info: dict[str, Any],
    ) -> AttackExecutionResult:
        """
        执行一次最小打击逻辑。

        这个函数只负责：
        1. 检查动作是否满足基础执行条件；
        2. 根据 UAV 攻击能力更新 Target 状态或防御力；
        3. 返回 AttackExecutionResult 给 RewardCalculator 使用。

        它不直接计算 reward。
        这样可以确保“动作执行规则”和“奖励函数设计”不会混在一起。
        """
        previous_defense = float(target.defense)
        target_was_active_before_action = target.is_active

        if not uav.is_active:
            info["event"] = "inactive_uav"
            info["message"] = f"UAV {uav.uav_id} is not active."

            return AttackExecutionResult(
                is_valid_action=False,
                invalid_reason="inactive_uav",
                target_was_active_before_action=target_was_active_before_action,
                previous_target_defense=previous_defense,
                remaining_target_defense=float(target.defense),
            )

        if not target.is_active:
            info["event"] = "inactive_target"
            info["message"] = f"Target {target.target_id} is not active."

            return AttackExecutionResult(
                is_valid_action=False,
                invalid_reason="inactive_target",
                target_was_active_before_action=target_was_active_before_action,
                previous_target_defense=previous_defense,
                remaining_target_defense=float(target.defense),
            )

        if uav.uav_type != UAVType.ATTACK:
            info["event"] = "non_attack_uav"
            info["message"] = (
                f"UAV {uav.uav_id} is {uav.uav_type.name}, "
                "not an attack UAV."
            )

            return AttackExecutionResult(
                is_valid_action=False,
                invalid_reason="non_attack_uav",
                target_was_active_before_action=target_was_active_before_action,
                previous_target_defense=previous_defense,
                remaining_target_defense=float(target.defense),
            )

        if not self.env_config.bounds.contains(uav.position):
            info["event"] = "uav_out_of_bounds"
            info["message"] = f"UAV {uav.uav_id} is outside battlefield bounds."

            return AttackExecutionResult(
                is_valid_action=False,
                invalid_reason="uav_out_of_bounds",
                target_was_active_before_action=target_was_active_before_action,
                previous_target_defense=previous_defense,
                remaining_target_defense=float(target.defense),
            )

        if not self.env_config.bounds.contains(target.position):
            info["event"] = "target_out_of_bounds"
            info["message"] = (
                f"Target {target.target_id} is outside battlefield bounds."
            )

            return AttackExecutionResult(
                is_valid_action=False,
                invalid_reason="target_out_of_bounds",
                target_was_active_before_action=target_was_active_before_action,
                previous_target_defense=previous_defense,
                remaining_target_defense=float(target.defense),
            )

        if not uav.can_reach(target):
            info["event"] = "out_of_range"
            info["message"] = (
                f"UAV {uav.uav_id} cannot reach Target {target.target_id}."
            )

            return AttackExecutionResult(
                is_valid_action=False,
                invalid_reason="out_of_range",
                target_was_active_before_action=target_was_active_before_action,
                previous_target_defense=previous_defense,
                remaining_target_defense=float(target.defense),
            )

        # 最小杀伤逻辑：
        # 若攻击能力覆盖目标防御力，则目标被摧毁；
        # 否则降低目标剩余防御力，表示目标受损但未被摧毁。
        if uav.attack_power >= target.defense:
            target.status = EntityStatus.DESTROYED

            info["event"] = "target_destroyed"
            info["message"] = (
                f"Target {target.target_id} destroyed by UAV {uav.uav_id}."
            )

            return AttackExecutionResult(
                is_valid_action=True,
                target_was_active_before_action=target_was_active_before_action,
                target_destroyed_after_action=True,
                target_damaged_after_action=False,
                previous_target_defense=previous_defense,
                remaining_target_defense=0.0,
            )

        target.defense -= uav.attack_power

        info["event"] = "target_damaged"
        info["message"] = (
            f"Target {target.target_id} damaged by UAV {uav.uav_id}. "
            f"Remaining defense: {target.defense:.2f}"
        )

        return AttackExecutionResult(
            is_valid_action=True,
            target_was_active_before_action=target_was_active_before_action,
            target_destroyed_after_action=False,
            target_damaged_after_action=True,
            previous_target_defense=previous_defense,
            remaining_target_defense=float(target.defense),
        )

    def _calculate_reward_for_invalid_action(
        self,
        invalid_reason: str,
        attacked_before: set[int],
    ) -> RewardBreakdown:
        """
        为无法解析到 UAV / Target 的非法动作构造奖励。

        例如 action 缺少 uav_id，或者给出的编号在当前战场状态中不存在。
        这类错误无法构造完整的 UAV / Target 上下文，因此使用 uav=None、target=None。
        """
        assert self.state is not None

        context = RewardContext(
            uav=None,
            target=None,
            current_step=self.state.current_step,
            current_time=self.current_time,
            is_valid_action=False,
            invalid_reason=invalid_reason,
            attacked_target_ids_before_action=attacked_before,
        )
        return self.reward_calculator.calculate(context)

    def _build_step_result(
        self,
        reward_breakdown: RewardBreakdown,
        info: dict[str, Any],
    ) -> StepResult:
        """
        统一构造 step() 返回结果。

        这样做可以保证所有动作分支，无论合法还是非法，都统一：
        1. 更新 episode_reward；
        2. 判断 done；
        3. 写入 reward_breakdown；
        4. 返回最新 observation。
        """
        self.episode_reward += reward_breakdown.total_reward

        info["reward_breakdown"] = reward_breakdown.to_dict()
        info["episode_reward"] = self.episode_reward
        info["attacked_target_ids"] = sorted(self.attacked_target_ids)

        done = self._is_done()

        return StepResult(
            observation=self._get_observation(),
            reward=reward_breakdown.total_reward,
            done=done,
            info=info,
        )

    def _get_observation(self) -> dict[str, Any]:
        """
        获取当前环境观测。

        当前返回的是便于人工调试的字典 observation。
        真正用于神经网络训练的固定长度向量，由 observation.py 中的
        ObservationBuilder 负责构造。
        """
        self._ensure_initialized()
        assert self.state is not None

        return {
            "env_name": self.env_config.name,
            "coordinate_mode": self.env_config.coordinate_mode,
            "coordinate_unit": self.env_config.coordinate_unit,
            "current_step": self.state.current_step,
            "current_time": self.current_time,
            "max_time": self.env_config.max_time,
            "time_step": self.env_config.time_step,
            "max_steps": self.env_config.max_steps,
            "uav_speed": self.env_config.uav_speed,
            "base_position": {
                "x": self.env_config.base_position.x,
                "y": self.env_config.base_position.y,
            },
            "battlefield_bounds": {
                "x_min": self.env_config.bounds.x_min,
                "x_max": self.env_config.bounds.x_max,
                "y_min": self.env_config.bounds.y_min,
                "y_max": self.env_config.bounds.y_max,
            },
            "active_uav_ids": [uav.uav_id for uav in self.state.active_uavs],
            "active_target_ids": [
                target.target_id for target in self.state.active_targets
            ],
            "num_active_uavs": len(self.state.active_uavs),
            "num_active_targets": len(self.state.active_targets),
            "num_attack_uavs": len(self.state.attack_uavs),
            "episode_reward": self.episode_reward,
            "attacked_target_ids": sorted(self.attacked_target_ids),
            "dynamic_events_enabled": self.env_config.enable_dynamic_events,
        }

    def _is_done(self) -> bool:
        """
        判断当前 episode 是否结束。

        当前结束条件：
        1. 所有目标都不再 active；
        2. 当前时间达到最大仿真时间；
        3. 没有可用攻击无人机。

        后面加入资源评估和动态支援机制后，还可以增加：
        - 攻击资源耗尽；
        - 任务完成率达到阈值；
        - 动态重分配失败；
        - 无可行合法动作。
        """
        self._ensure_initialized()
        assert self.state is not None

        if len(self.state.active_targets) == 0:
            return True

        if self.current_time >= self.env_config.max_time:
            return True

        if len(self.state.attack_uavs) == 0:
            return True

        return False

    def _ensure_initialized(self) -> None:
        """
        确保环境已经初始化。

        使用强化学习环境时，必须先调用 reset()，再调用 step()。
        这里提前检查可以避免出现更难排查的 NoneType 错误。
        """
        if self.state is None:
            raise EnvironmentError(
                "Environment is not initialized. "
                "Call env.reset() before env.step()."
            )

    def get_state_copy(self) -> BattlefieldState:
        """
        返回当前战场状态的深拷贝。

        外部调试、可视化、动作空间构建可以读取这份副本，
        但不会误改环境内部真实 state。
        """
        self._ensure_initialized()
        assert self.state is not None

        return deepcopy(self.state)