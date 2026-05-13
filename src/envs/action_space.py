"""envs 数据模块中的动作space实现。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from uav_dynamic_task_allocation.utils.config import get_config_value

from uav_dynamic_task_allocation.core.entities import (
    BattlefieldState,
    Target,
    UAV,
    UAVType,
)
from uav_dynamic_task_allocation.envs.env_config import EnvConfig


class ActionSpaceError(Exception):
    """动作空间构建与动作解析过程中的自定义错误。"""


@dataclass(frozen=True)
class ActionValidationResult:
    """
    动作合法性检查结果。

    这个对象的作用不是只返回 True / False，而是同时记录“不合法原因”。
    这样做对调试强化学习环境非常重要，因为训练初期智能体会频繁选择无效动作，
    如果没有清楚的 reason，后面很难判断 reward 异常到底是环境问题还是策略问题。
    """

    # is_valid: isvalid。
    is_valid: bool
    # reason: reason 数据。
    reason: str = "valid"


@dataclass(frozen=True)
class UAVTargetAction:
    """
    UAV-Target 基础打击动作。

    该动作表示：选择一架 UAV 对一个目标执行一次打击尝试。

    当前动作空间服务于第 4 章“打击次序决策”主线：
    DQN 输出 action_id，action_space 将其映射为 UAVTargetAction，
    环境再执行该动作并返回 reward、done 和新的 observation。

    注意：
    这里暂时不表达“目标分群动作”。论文第 3 章 PPO 分群里的
    源群-目标-新归属分群三层动作空间，会在后续 clustering/ppo 模块单独实现。
    """

    # action_id: 动作编号。
    action_id: int
    # uav_id: 无人机编号。
    uav_id: int
    # target_id: 目标编号。
    target_id: int
    # distance: distance 数据。
    distance: float
    # target_value: 目标数值。
    target_value: float
    # is_valid: isvalid。
    is_valid: bool
    # invalid_reason: invalidreason。
    invalid_reason: str = "valid"

    def to_env_action(self) -> dict[str, int]:
        """
        转换为 DroneBattleEnv.step() 当前接受的动作格式。

        这样做是为了把“强化学习动作编号”和“环境可解释动作”解耦。
        后面 DQN 只需要处理 action_id，而环境仍然可以接收清晰的字典动作。
        """
        return {
            "uav_id": self.uav_id,
            "target_id": self.target_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        转换为普通字典，主要用于日志记录、调试输出和保存实验结果。
        """
        return {
            "action_id": self.action_id,
            "uav_id": self.uav_id,
            "target_id": self.target_id,
            "distance": self.distance,
            "target_value": self.target_value,
            "is_valid": self.is_valid,
            "invalid_reason": self.invalid_reason,
        }


class UAVTargetActionSpace:
    """
    UAV-Target 动作空间。

    这个类的职责是：
    1. 根据当前 BattlefieldState 生成所有候选 UAV-Target 动作；
    2. 根据基础规则筛选合法动作；
    3. 建立 action_id 到 UAVTargetAction 的映射；
    4. 为后续 DQN/PPO/SAC 提供统一动作接口。

    为什么不直接在 DroneBattleEnv 里面写这些逻辑？
    因为环境应该负责“执行动作和更新状态”，而动作空间应该负责“动作枚举和合法性过滤”。
    如果全部写在 env.step() 里，后面训练、评估、调试都会越来越乱。
    """

    def __init__(
        self,
        env_config: EnvConfig,
        include_invalid_actions: bool = False,
    ) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            env_config: env_config 参数，类型为 EnvConfig。
            include_invalid_actions: include_invalid_actions 参数，类型为 bool。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        # env_config: 环境配置。
        self.env_config = env_config
        # include_invalid_actions: includeinvalid动作集合。
        self.include_invalid_actions = include_invalid_actions

        # actions: 动作集合。
        self.actions: list[UAVTargetAction] = []
        # _action_id_to_action: 动作编号to动作。
        self._action_id_to_action: dict[int, UAVTargetAction] = {}

    def build(self, state: BattlefieldState) -> list[UAVTargetAction]:
        """
        根据当前战场状态构建动作空间。

        这个函数每次根据 state 重新生成动作空间，而不是只在环境初始化时生成一次。
        原因是动态战场中 UAV 和 Target 状态会变化：
        - 目标可能被摧毁；
        - UAV 可能被击毁；
        - 新目标可能出现；
        - 原目标可能消失。

        因此动作空间必须能够跟随 state 动态更新。
        """
        self.actions = []
        self._action_id_to_action = {}

        action_id = 0

        for uav in state.uavs:
            for target in state.targets:
                validation = self.validate_action(
                    state=state,
                    uav=uav,
                    target=target,
                )

                if not validation.is_valid and not self.include_invalid_actions:
                    continue

                distance = uav.distance_to_target(target)
                action = UAVTargetAction(
                    action_id=action_id,
                    uav_id=uav.uav_id,
                    target_id=target.target_id,
                    distance=distance,
                    target_value=target.value_score,
                    is_valid=validation.is_valid,
                    invalid_reason=validation.reason,
                )

                self.actions.append(action)
                self._action_id_to_action[action_id] = action
                action_id += 1

        return self.actions

    def validate_action(
        self,
        state: BattlefieldState,
        uav: UAV,
        target: Target,
    ) -> ActionValidationResult:
        """
        检查一个 UAV-Target 动作是否合法。

        当前版本只实现“基础动作合法性规则”，这些规则来自环境最小运行所需条件：
        1. UAV 必须仍然 active；
        2. Target 必须仍然 active；
        3. UAV 必须是攻击无人机；
        4. UAV 和 Target 必须在战场边界内；
        5. UAV 必须能够覆盖目标。

        后面我们会把更复杂的约束拆到 core/constraints.py，例如：
        - 资源约束；
        - 杀伤链约束；
        - 通信约束；
        - 多无人机协同约束；
        - 时间窗口约束；
        - 动态支援约束。

        现在不要把这些全部塞进 action_space，否则动作空间会过早变复杂。
        """
        if uav not in state.uavs:
            return ActionValidationResult(
                is_valid=False,
                reason="uav_not_in_state",
            )

        if target not in state.targets:
            return ActionValidationResult(
                is_valid=False,
                reason="target_not_in_state",
            )

        if not uav.is_active:
            return ActionValidationResult(
                is_valid=False,
                reason="inactive_uav",
            )

        if not target.is_active:
            return ActionValidationResult(
                is_valid=False,
                reason="inactive_target",
            )

        if uav.uav_type != UAVType.ATTACK:
            return ActionValidationResult(
                is_valid=False,
                reason="non_attack_uav",
            )

        if not self.env_config.bounds.contains(uav.position):
            return ActionValidationResult(
                is_valid=False,
                reason="uav_out_of_bounds",
            )

        if not self.env_config.bounds.contains(target.position):
            return ActionValidationResult(
                is_valid=False,
                reason="target_out_of_bounds",
            )

        if not uav.can_reach(target):
            return ActionValidationResult(
                is_valid=False,
                reason="out_of_range",
            )

        return ActionValidationResult(is_valid=True)

    def get_action(self, action_id: int) -> UAVTargetAction:
        """
        根据 action_id 获取动作对象。

        DQN 网络输出的是动作编号，所以训练代码需要通过这个函数把编号转换成
        可解释的 UAVTargetAction。
        """
        if action_id not in self._action_id_to_action:
            raise ActionSpaceError(
                f"Invalid action_id: {action_id}. "
                f"Available action ids: {list(self._action_id_to_action.keys())[:10]}..."
            )

        return self._action_id_to_action[action_id]

    def get_env_action(self, action_id: int) -> dict[str, int]:
        """
        根据 action_id 获取 DroneBattleEnv.step() 可直接执行的动作字典。
        """
        return self.get_action(action_id).to_env_action()

    def valid_actions(self) -> list[UAVTargetAction]:
        """
        返回当前动作空间中的合法动作。

        如果 include_invalid_actions=False，则 self.actions 本身就全是合法动作；
        如果 include_invalid_actions=True，则这里会进一步过滤。
        """
        return [action for action in self.actions if action.is_valid]

    def invalid_actions(self) -> list[UAVTargetAction]:
        """
        返回当前动作空间中的非法动作，主要用于调试环境和分析动作过滤规则。
        """
        return [action for action in self.actions if not action.is_valid]

    def size(self) -> int:
        """
        返回当前动作空间大小。

        对 DQN 来说，这个值将对应输出层维度。
        不过要注意：动态环境中动作空间大小可能变化。
        后面正式 DQN 时，我们需要进一步讨论是使用固定动作空间，还是使用动作 mask。
        """
        return len(self.actions)

    def summary(self) -> dict[str, Any]:
        """
        返回动作空间统计信息，方便日志记录和调试。
        """
        valid_count = len(self.valid_actions())
        invalid_count = len(self.invalid_actions())

        invalid_reason_counts: dict[str, int] = {}
        for action in self.invalid_actions():
            invalid_reason_counts[action.invalid_reason] = (
                invalid_reason_counts.get(action.invalid_reason, 0) + 1
            )

        return {
            "total_actions": len(self.actions),
            "valid_actions": valid_count,
            "invalid_actions": invalid_count,
            "include_invalid_actions": self.include_invalid_actions,
            "invalid_reason_counts": invalid_reason_counts,
        }

@dataclass(frozen=True)
class ActionSpaceConfig:
    """
    固定动作空间配置。

    该配置用于控制 DQN 输出层的动作维度。
    DQN 要求输出维度固定，因此不能每一步都根据合法动作数量动态改变输出层。
    本项目采用 slot-based 固定动作空间：

        action_id = uav_slot * max_targets + target_slot

    其中 uav_slot 和 target_slot 对应 observation 中 UAV / Target 的固定位置。
    这样可以保证 observation 编码、action 编码和 action mask 使用同一套索引体系。
    """

    # mode: mode 数据。
    mode: str = "fixed_slot"
    # max_uavs: 最大值uavs。
    max_uavs: int = 50
    # max_targets: 最大值目标集合。
    max_targets: int = 120
    # attack_uav_only: 攻击无人机only。
    attack_uav_only: bool = True
    # require_reachable: requirereachable。
    require_reachable: bool = True
    # allow_noop_when_no_valid_action: allownoopwhennovalid动作。
    allow_noop_when_no_valid_action: bool = False

    def validate(self) -> None:
        """检查动作空间配置是否合法。"""
        if self.mode not in {"fixed_slot"}:
            raise ActionSpaceError(
                f"Invalid action space mode: {self.mode}. "
                "Currently supported mode: fixed_slot."
            )

        if self.max_uavs <= 0:
            raise ActionSpaceError("max_uavs must be positive.")

        if self.max_targets <= 0:
            raise ActionSpaceError("max_targets must be positive.")


@dataclass(frozen=True)
class FixedActionDecodeResult:
    """
    固定动作编号解码结果。

    action_id 本身只是一个整数，不能直接说明要让哪架 UAV 打击哪个目标。
    该对象用于把 action_id 解码成：
    - uav_slot；
    - target_slot；
    - 实际 uav_id；
    - 实际 target_id；
    - 是否合法；
    - 不合法原因。

    这样训练和调试时就可以清楚知道，DQN 选择的动作到底是什么意思。
    """

    # action_id: 动作编号。
    action_id: int
    # uav_slot: 无人机slot。
    uav_slot: int
    # target_slot: 目标slot。
    target_slot: int
    # uav_id: 无人机编号。
    uav_id: int | None
    # target_id: 目标编号。
    target_id: int | None
    # is_valid: isvalid。
    is_valid: bool
    # invalid_reason: invalidreason。
    invalid_reason: str = "valid"
    # distance: distance 数据。
    distance: float | None = None
    # target_value: 目标数值。
    target_value: float | None = None

    def to_env_action(self) -> dict[str, int]:
        """
        转换为 DroneBattleEnv.step() 可执行的动作。

        只有合法动作才能转换。如果非法动作也强行执行，环境只能收到缺失或错误编号，
        这会让训练日志变得混乱，所以这里直接阻止非法动作转换。
        """
        if not self.is_valid or self.uav_id is None or self.target_id is None:
            raise ActionSpaceError(
                f"Cannot convert invalid action to env action: {self}"
            )

        return {
            "uav_id": self.uav_id,
            "target_id": self.target_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，便于日志输出和保存实验结果。"""
        return {
            "action_id": self.action_id,
            "uav_slot": self.uav_slot,
            "target_slot": self.target_slot,
            "uav_id": self.uav_id,
            "target_id": self.target_id,
            "is_valid": self.is_valid,
            "invalid_reason": self.invalid_reason,
            "distance": self.distance,
            "target_value": self.target_value,
        }


class FixedUAVTargetActionSpace:
    """
    固定 UAV-Target 动作空间。

    该类是后续 DQN 的动作接口基础。与前面的动态动作空间不同，
    固定动作空间不会因为当前合法动作数量变化而改变总维度。

    固定动作空间的核心规则是：

        action_id = uav_slot * max_targets + target_slot

    例如 max_targets = 120：
        action_id = 0   -> uav_slot=0, target_slot=0
        action_id = 1   -> uav_slot=0, target_slot=1
        action_id = 120 -> uav_slot=1, target_slot=0

    这样做的好处：
    1. DQN 输出层维度固定；
    2. 动态事件发生后，只需要更新 action mask；
    3. observation 中的 UAV / Target slot 与 action space 保持一致；
    4. 非法动作不会消失，而是通过 mask 屏蔽。
    """

    def __init__(
        self,
        env_config: EnvConfig,
        action_space_config: ActionSpaceConfig,
    ) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            env_config: env_config 参数，类型为 EnvConfig。
            action_space_config: action_space_config 参数，类型为 ActionSpaceConfig。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        # env_config: 环境配置。
        self.env_config = env_config
        # config: 配置。
        self.config = action_space_config
        self.config.validate()

    @property
    def size(self) -> int:
        """
        固定动作空间大小。

        该值后续会作为 DQN 网络输出层维度。
        """
        return self.config.max_uavs * self.config.max_targets

    def encode_action_id(self, uav_slot: int, target_slot: int) -> int:
        """
        根据 UAV slot 和 Target slot 编码 action_id。

        slot 不是原始 uav_id / target_id，而是对象在固定 observation 序列中的位置。
        """
        self._validate_slot_indices(uav_slot, target_slot)
        return uav_slot * self.config.max_targets + target_slot

    def decode_action_id(
        self,
        state: BattlefieldState,
        action_id: int,
    ) -> FixedActionDecodeResult:
        """
        将固定 action_id 解码为具体 UAV-Target 动作。

        该函数会同时判断动作是否合法。
        如果 action_id 对应的是 padding slot，或者 UAV / Target 状态不可用，
        返回结果中的 is_valid 会是 False，并给出 invalid_reason。
        """
        if action_id < 0 or action_id >= self.size:
            raise ActionSpaceError(
                f"action_id out of range: {action_id}. "
                f"Valid range: [0, {self.size - 1}]"
            )

        uav_slot = action_id // self.config.max_targets
        target_slot = action_id % self.config.max_targets

        uav = self._get_uav_by_slot(state, uav_slot)
        target = self._get_target_by_slot(state, target_slot)

        if uav is None:
            return FixedActionDecodeResult(
                action_id=action_id,
                uav_slot=uav_slot,
                target_slot=target_slot,
                uav_id=None,
                target_id=target.target_id if target is not None else None,
                is_valid=False,
                invalid_reason="padding_uav_slot",
            )

        if target is None:
            return FixedActionDecodeResult(
                action_id=action_id,
                uav_slot=uav_slot,
                target_slot=target_slot,
                uav_id=uav.uav_id,
                target_id=None,
                is_valid=False,
                invalid_reason="padding_target_slot",
            )

        validation = self.validate_uav_target_action(
            state=state,
            uav=uav,
            target=target,
        )

        distance = uav.distance_to_target(target)
        target_value = target.value_score

        return FixedActionDecodeResult(
            action_id=action_id,
            uav_slot=uav_slot,
            target_slot=target_slot,
            uav_id=uav.uav_id,
            target_id=target.target_id,
            is_valid=validation.is_valid,
            invalid_reason=validation.reason,
            distance=distance,
            target_value=target_value,
        )

    def build_action_mask(self, state: BattlefieldState) -> np.ndarray:
        """
        构造固定长度 action mask。

        mask 的长度等于固定动作空间大小：
            max_uavs * max_targets

        每个位置的含义：
            1.0 表示该 action_id 当前合法；
            0.0 表示该 action_id 当前非法。

        DQN 选择动作时，可以将非法动作对应的 Q 值设为极小值，
        从而避免智能体选择不可执行动作。
        """
        mask = np.zeros(self.size, dtype=np.float32)

        for action_id in range(self.size):
            decoded = self.decode_action_id(state, action_id)
            if decoded.is_valid:
                mask[action_id] = 1.0

        return mask

    def get_valid_action_ids(self, state: BattlefieldState) -> list[int]:
        """
        返回当前状态下所有合法 action_id。
        """
        mask = self.build_action_mask(state)
        return np.where(mask == 1.0)[0].astype(int).tolist()

    def get_invalid_action_ids(self, state: BattlefieldState) -> list[int]:
        """
        返回当前状态下所有非法 action_id。
        """
        mask = self.build_action_mask(state)
        return np.where(mask == 0.0)[0].astype(int).tolist()

    def select_first_valid_action(
        self,
        state: BattlefieldState,
    ) -> FixedActionDecodeResult:
        """
        选择第一个合法动作，主要用于测试和 baseline 调试。

        后续 DQN 训练时不会用这个函数选动作，
        而是由 Q 网络输出 Q 值，再结合 action mask 选择动作。
        """
        valid_action_ids = self.get_valid_action_ids(state)

        if not valid_action_ids:
            raise ActionSpaceError(
                "No valid action available in current state."
            )

        return self.decode_action_id(state, valid_action_ids[0])

    def validate_uav_target_action(
        self,
        state: BattlefieldState,
        uav: UAV,
        target: Target,
    ) -> ActionValidationResult:
        """
        检查 UAV-Target 动作是否合法。

        这里的规则要和 DroneBattleEnv._execute_attack() 保持一致。
        action_space 负责提前屏蔽非法动作；
        env 负责在真正执行时再次防御性检查。

        两层检查不是重复，而是工程上的安全设计：
        - action_space 用于训练前筛选；
        - env 用于运行时兜底。
        """
        if uav not in state.uavs:
            return ActionValidationResult(False, "uav_not_in_state")

        if target not in state.targets:
            return ActionValidationResult(False, "target_not_in_state")

        if not uav.is_active:
            return ActionValidationResult(False, "inactive_uav")

        if not target.is_active:
            return ActionValidationResult(False, "inactive_target")

        if self.config.attack_uav_only and uav.uav_type != UAVType.ATTACK:
            return ActionValidationResult(False, "non_attack_uav")

        if not self.env_config.bounds.contains(uav.position):
            return ActionValidationResult(False, "uav_out_of_bounds")

        if not self.env_config.bounds.contains(target.position):
            return ActionValidationResult(False, "target_out_of_bounds")

        if self.config.require_reachable and not uav.can_reach(target):
            return ActionValidationResult(False, "out_of_range")

        return ActionValidationResult(True, "valid")

    def summarize_mask(self, state: BattlefieldState) -> dict[str, Any]:
        """
        汇总 action mask 信息。

        这个函数用于日志和调试，帮助我们判断：
        - 固定动作空间总共有多大；
        - 当前有多少动作合法；
        - 非法动作主要因为什么被屏蔽。
        """
        mask = self.build_action_mask(state)

        invalid_reason_counts: dict[str, int] = {}

        for action_id in range(self.size):
            decoded = self.decode_action_id(state, action_id)
            if not decoded.is_valid:
                invalid_reason_counts[decoded.invalid_reason] = (
                    invalid_reason_counts.get(decoded.invalid_reason, 0) + 1
                )

        valid_count = int(mask.sum())
        invalid_count = int(self.size - valid_count)

        return {
            "fixed_action_space_size": self.size,
            "valid_action_count": valid_count,
            "invalid_action_count": invalid_count,
            "valid_action_ratio": valid_count / max(self.size, 1),
            "invalid_reason_counts": invalid_reason_counts,
        }

    def _get_uav_by_slot(
        self,
        state: BattlefieldState,
        uav_slot: int,
    ) -> UAV | None:
        """
        根据固定 UAV slot 获取 UAV 对象。

        这里使用 state.uavs 的顺序，与 ObservationBuilder 中 UAV 序列顺序保持一致。
        如果 uav_slot 超过当前真实 UAV 数量，则说明该 slot 是 padding。
        """
        if uav_slot >= len(state.uavs):
            return None

        return state.uavs[uav_slot]

    def _get_target_by_slot(
        self,
        state: BattlefieldState,
        target_slot: int,
    ) -> Target | None:
        """
        根据固定 Target slot 获取 Target 对象。

        这里使用 state.targets 的顺序，与 ObservationBuilder 中 Target 序列顺序保持一致。
        如果 target_slot 超过当前真实 Target 数量，则说明该 slot 是 padding。
        """
        if target_slot >= len(state.targets):
            return None

        return state.targets[target_slot]

    def _validate_slot_indices(
        self,
        uav_slot: int,
        target_slot: int,
    ) -> None:
        """
        检查 slot 编号是否位于固定动作空间范围内。
        """
        if not 0 <= uav_slot < self.config.max_uavs:
            raise ActionSpaceError(
                f"Invalid uav_slot: {uav_slot}. "
                f"Valid range: [0, {self.config.max_uavs - 1}]"
            )

        if not 0 <= target_slot < self.config.max_targets:
            raise ActionSpaceError(
                f"Invalid target_slot: {target_slot}. "
                f"Valid range: [0, {self.config.max_targets - 1}]"
            )


def load_action_space_config(config: dict[str, Any]) -> ActionSpaceConfig:
    """
    从项目总配置中读取动作空间配置。

    如果 action_space.max_uavs / max_targets 没有单独配置，
    则默认读取 observation.max_uavs / observation.max_targets，
    保证 observation slot 和 action slot 对齐。
    """
    max_uavs = int(
        get_config_value(
            config,
            "action_space.max_uavs",
            default=get_config_value(config, "observation.max_uavs", default=50),
        )
    )
    max_targets = int(
        get_config_value(
            config,
            "action_space.max_targets",
            default=get_config_value(
                config,
                "observation.max_targets",
                default=120,
            ),
        )
    )

    action_space_config = ActionSpaceConfig(
        mode=str(
            get_config_value(config, "action_space.mode", default="fixed_slot")
        ),
        max_uavs=max_uavs,
        max_targets=max_targets,
        attack_uav_only=bool(
            get_config_value(
                config,
                "action_space.attack_uav_only",
                default=True,
            )
        ),
        require_reachable=bool(
            get_config_value(
                config,
                "action_space.require_reachable",
                default=True,
            )
        ),
        allow_noop_when_no_valid_action=bool(
            get_config_value(
                config,
                "action_space.allow_noop_when_no_valid_action",
                default=False,
            )
        ),
    )
    action_space_config.validate()

    return action_space_config
