from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from uav_dynamic_task_allocation.core.entities import Target, UAV
from uav_dynamic_task_allocation.envs.env_config import EnvConfig
from uav_dynamic_task_allocation.utils.config import get_config_value


class RewardError(Exception):
    """奖励函数配置、计算和调试过程中的自定义错误。"""


@dataclass(frozen=True)
class RewardConfig:
    """
    奖励函数配置。

    该配置用于控制奖励函数各组成部分的权重和惩罚值。
    论文中的基础奖励结构为：
        R = R_dist + R_rep + R_pri

    但在工程实现中，还需要处理非法动作、越界动作、非攻击无人机动作等情况。
    因此这里把论文奖励项和工程惩罚项都配置化，方便后续实验调参。
    """

    mode: str = "paper"

    use_normalized_distance: bool = True
    distance_normalizer: str | float = "auto"
    distance_weight: float = 1.0

    repetition_penalty: float = -10.0

    priority_weight: float = 1.0
    target_value_normalizer: float = 100.0

    invalid_action_penalty: float = -10.0
    inactive_uav_penalty: float = -5.0
    inactive_target_penalty: float = -10.0
    non_attack_uav_penalty: float = -2.0
    out_of_range_penalty: float = -1.0
    out_of_bounds_penalty: float = -2.0

    destroy_bonus: float = 0.0
    damage_reward_weight: float = 0.0

    def validate(self) -> None:
        """检查奖励配置是否合法，避免训练时出现不可解释的 reward 异常。"""
        if self.mode not in {"paper"}:
            raise RewardError(
                f"Invalid reward mode: {self.mode}. "
                "Currently supported mode: paper."
            )

        if self.distance_weight < 0:
            raise RewardError("distance_weight should be non-negative.")

        if self.priority_weight < 0:
            raise RewardError("priority_weight should be non-negative.")

        if self.target_value_normalizer <= 0:
            raise RewardError("target_value_normalizer must be positive.")

        if self.damage_reward_weight < 0:
            raise RewardError("damage_reward_weight should be non-negative.")


@dataclass
class RewardContext:
    """
    单步奖励计算上下文。

    奖励函数不能只看 UAV 和 Target，还需要知道动作执行前后的状态。
    例如，重复打击惩罚要判断目标在动作前是否已经被打击；
    damage reward 要知道目标防御力减少了多少；
    priority reward 要知道该目标在第几步被选择。

    因此这里用 RewardContext 把 reward 所需信息集中传入，
    避免 RewardCalculator 直接依赖 DroneBattleEnv 内部实现细节。
    """

    uav: UAV | None
    target: Target | None

    current_step: int
    current_time: float

    is_valid_action: bool = True
    invalid_reason: str | None = None

    target_was_active_before_action: bool = True
    target_destroyed_after_action: bool = False
    target_damaged_after_action: bool = False

    previous_target_defense: float | None = None
    remaining_target_defense: float | None = None

    attacked_target_ids_before_action: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class RewardBreakdown:
    """
    奖励分解结果。

    不只返回 total_reward，而是把每一项奖励都保存下来。
    这样训练日志中可以清楚看到：
        - 模型是不是被距离惩罚压制；
        - 重复打击惩罚是否太大；
        - 优先级奖励是否起作用；
        - 非法动作是不是太多。

    这对调试强化学习非常重要。
    """

    total_reward: float

    distance_reward: float = 0.0
    repetition_penalty: float = 0.0
    priority_reward: float = 0.0

    invalid_action_penalty: float = 0.0
    destroy_bonus: float = 0.0
    damage_reward: float = 0.0

    reason: str = "valid"

    def to_dict(self) -> dict[str, float | str]:
        """转换为字典，方便 logger 输出和保存到 CSV。"""
        return {
            "total_reward": self.total_reward,
            "distance_reward": self.distance_reward,
            "repetition_penalty": self.repetition_penalty,
            "priority_reward": self.priority_reward,
            "invalid_action_penalty": self.invalid_action_penalty,
            "destroy_bonus": self.destroy_bonus,
            "damage_reward": self.damage_reward,
            "reason": self.reason,
        }


class RewardCalculator:
    """
    UAV-Target 打击任务奖励计算器。

    该类将论文中的奖励函数结构工程化：
        R = R_dist + R_rep + R_pri

    其中：
    - R_dist：负距离奖励，鼓励选择路径代价较低的目标；
    - R_rep：重复打击惩罚，避免重复攻击已处理目标；
    - R_pri：优先级奖励，鼓励尽早打击高价值目标。

    同时，为了保证环境能够稳定训练，本类还处理非法动作惩罚。
    这些非法动作惩罚不是论文核心贡献，而是工程环境必须具备的安全反馈机制。
    """

    def __init__(
        self,
        env_config: EnvConfig,
        reward_config: RewardConfig,
    ) -> None:
        self.env_config = env_config
        self.reward_config = reward_config
        self.reward_config.validate()

        self.distance_normalizer = self._resolve_distance_normalizer()

    def calculate(self, context: RewardContext) -> RewardBreakdown:
        """
        计算单步奖励。

        计算顺序：
        1. 如果是非法动作，优先返回非法动作惩罚；
        2. 如果是重复打击，加入重复打击惩罚；
        3. 计算负距离奖励；
        4. 计算目标优先级奖励；
        5. 根据配置决定是否加入摧毁奖励和损伤奖励。
        """
        if not context.is_valid_action:
            return self._calculate_invalid_action_reward(context)

        if context.uav is None or context.target is None:
            return RewardBreakdown(
                total_reward=self.reward_config.invalid_action_penalty,
                invalid_action_penalty=self.reward_config.invalid_action_penalty,
                reason="missing_uav_or_target",
            )

        distance_reward = self._calculate_distance_reward(
            uav=context.uav,
            target=context.target,
        )

        repetition_penalty = self._calculate_repetition_penalty(context)

        priority_reward = self._calculate_priority_reward(
            target=context.target,
            current_step=context.current_step,
        )

        destroy_bonus = (
            self.reward_config.destroy_bonus
            if context.target_destroyed_after_action
            else 0.0
        )

        damage_reward = self._calculate_damage_reward(context)

        total_reward = (
            distance_reward
            + repetition_penalty
            + priority_reward
            + destroy_bonus
            + damage_reward
        )

        reason = "valid"
        if repetition_penalty != 0.0:
            reason = "repeated_target"

        return RewardBreakdown(
            total_reward=total_reward,
            distance_reward=distance_reward,
            repetition_penalty=repetition_penalty,
            priority_reward=priority_reward,
            destroy_bonus=destroy_bonus,
            damage_reward=damage_reward,
            reason=reason,
        )

    def _calculate_invalid_action_reward(
        self,
        context: RewardContext,
    ) -> RewardBreakdown:
        """
        计算非法动作惩罚。

        这里把不同非法原因映射到不同惩罚值。
        这样比所有非法动作都给 -10 更细致：
        - 格式错误、实体不存在属于严重错误；
        - 超出距离属于可学习的战术错误；
        - 非攻击无人机执行攻击属于任务类型错误；
        - 重复打击已处理目标对应论文中的高惩罚。
        """
        reason = context.invalid_reason or "invalid_action"
        penalty = self._get_invalid_action_penalty(reason)

        return RewardBreakdown(
            total_reward=penalty,
            invalid_action_penalty=penalty,
            reason=reason,
        )

    def _get_invalid_action_penalty(self, reason: str) -> float:
        """
        根据非法动作原因返回对应惩罚。

        如果后面新增约束，例如通信约束、载荷约束、资源不足约束，
        可以在这里继续添加映射。
        """
        if reason in {"invalid_action_format", "entity_not_found"}:
            return self.reward_config.invalid_action_penalty

        if reason == "inactive_uav":
            return self.reward_config.inactive_uav_penalty

        if reason == "inactive_target":
            return self.reward_config.inactive_target_penalty

        if reason == "non_attack_uav":
            return self.reward_config.non_attack_uav_penalty

        if reason == "out_of_range":
            return self.reward_config.out_of_range_penalty

        if reason in {"uav_out_of_bounds", "target_out_of_bounds"}:
            return self.reward_config.out_of_bounds_penalty

        return self.reward_config.invalid_action_penalty

    def _calculate_distance_reward(self, uav: UAV, target: Target) -> float:
        """
        计算负距离奖励 R_dist。

        论文中使用负距离作为路径代价项。由于真实战场坐标尺度较大，
        这里默认使用归一化距离，避免距离项过大导致优先级奖励失效。
        """
        distance = uav.distance_to_target(target)

        if self.reward_config.use_normalized_distance:
            distance_value = self._safe_divide(
                distance,
                self.distance_normalizer,
            )
        else:
            distance_value = distance

        return -self.reward_config.distance_weight * distance_value

    def _calculate_repetition_penalty(
        self,
        context: RewardContext,
    ) -> float:
        """
        计算重复打击惩罚 R_rep。

        重复打击有两种常见情况：
        1. 目标在动作前已经不是 active；
        2. 目标编号已经出现在 attacked_target_ids_before_action 中。

        当前环境里，目标被摧毁后 status 会改变；后续训练时也会维护 attacked history。
        两种方式都支持，是为了让 reward 模块更稳。
        """
        if context.target is None:
            return 0.0

        target_id = context.target.target_id

        if not context.target_was_active_before_action:
            return self.reward_config.repetition_penalty

        if target_id in context.attacked_target_ids_before_action:
            return self.reward_config.repetition_penalty

        return 0.0

    def _calculate_priority_reward(
        self,
        target: Target,
        current_step: int,
    ) -> float:
        """
        计算优先级奖励 R_pri。

        论文中优先级奖励的思想是：
        高优先级目标越早被打击，奖励越高。

        工程实现中采用：
            priority_reward = priority_weight * normalized_value / rank

        其中 rank 使用 current_step + 1，避免第 0 步除以 0。
        """
        rank = max(current_step, 1)
        normalized_value = self._safe_divide(
            target.value_score,
            self.reward_config.target_value_normalizer,
        )

        return self.reward_config.priority_weight * normalized_value / rank

    def _calculate_damage_reward(
        self,
        context: RewardContext,
    ) -> float:
        """
        计算目标受损奖励。

        该项是工程扩展项，默认权重为 0，不影响论文基础奖励结构。
        后续如果发现模型只关注距离、不愿意攻击高防御目标，
        可以适当开启该项，用防御力削减比例作为额外学习信号。
        """
        if self.reward_config.damage_reward_weight == 0.0:
            return 0.0

        if not context.target_damaged_after_action:
            return 0.0

        if context.previous_target_defense is None:
            return 0.0

        if context.remaining_target_defense is None:
            return 0.0

        defense_reduction = max(
            context.previous_target_defense - context.remaining_target_defense,
            0.0,
        )

        damage_ratio = self._safe_divide(
            defense_reduction,
            max(context.previous_target_defense, 1e-8),
        )

        return self.reward_config.damage_reward_weight * damage_ratio

    def _resolve_distance_normalizer(self) -> float:
        """
        解析距离归一化尺度。

        auto 模式下使用战场边界对角线作为归一化尺度。
        这样 toy 环境和真实战场环境都能得到相对合理的距离 reward。
        """
        if self.reward_config.distance_normalizer == "auto":
            x_span = self.env_config.bounds.x_max - self.env_config.bounds.x_min
            y_span = self.env_config.bounds.y_max - self.env_config.bounds.y_min
            return sqrt(x_span**2 + y_span**2)

        value = float(self.reward_config.distance_normalizer)
        if value <= 0:
            raise RewardError("distance_normalizer must be positive.")

        return value

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """安全除法，避免除零导致训练中断。"""
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)


def load_reward_config(config: dict[str, Any]) -> RewardConfig:
    """
    从项目总配置中读取 reward 配置。

    将 reward 参数放到 YAML 中，可以保证后续实验只需要改配置，
    不需要频繁修改 Python 代码。
    """
    reward_config = RewardConfig(
        mode=str(get_config_value(config, "reward.mode", default="paper")),
        use_normalized_distance=bool(
            get_config_value(
                config,
                "reward.use_normalized_distance",
                default=True,
            )
        ),
        distance_normalizer=get_config_value(
            config,
            "reward.distance_normalizer",
            default="auto",
        ),
        distance_weight=float(
            get_config_value(config, "reward.distance_weight", default=1.0)
        ),
        repetition_penalty=float(
            get_config_value(config, "reward.repetition_penalty", default=-10.0)
        ),
        priority_weight=float(
            get_config_value(config, "reward.priority_weight", default=1.0)
        ),
        target_value_normalizer=float(
            get_config_value(
                config,
                "reward.target_value_normalizer",
                default=100.0,
            )
        ),
        invalid_action_penalty=float(
            get_config_value(
                config,
                "reward.invalid_action_penalty",
                default=-10.0,
            )
        ),
        inactive_uav_penalty=float(
            get_config_value(
                config,
                "reward.inactive_uav_penalty",
                default=-5.0,
            )
        ),
        inactive_target_penalty=float(
            get_config_value(
                config,
                "reward.inactive_target_penalty",
                default=-10.0,
            )
        ),
        non_attack_uav_penalty=float(
            get_config_value(
                config,
                "reward.non_attack_uav_penalty",
                default=-2.0,
            )
        ),
        out_of_range_penalty=float(
            get_config_value(
                config,
                "reward.out_of_range_penalty",
                default=-1.0,
            )
        ),
        out_of_bounds_penalty=float(
            get_config_value(
                config,
                "reward.out_of_bounds_penalty",
                default=-2.0,
            )
        ),
        destroy_bonus=float(
            get_config_value(config, "reward.destroy_bonus", default=0.0)
        ),
        damage_reward_weight=float(
            get_config_value(
                config,
                "reward.damage_reward_weight",
                default=0.0,
            )
        ),
    )
    reward_config.validate()

    return reward_config