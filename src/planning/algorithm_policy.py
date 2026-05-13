from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from uav_dynamic_task_allocation.core.contracts import (
    MissionEventType,
    PipelineStage,
)
from uav_dynamic_task_allocation.utils.config import get_config_value


class AlgorithmPolicyError(Exception):
    """算法策略配置、查询和阶段决策过程中的自定义错误。"""


class ScenarioMode(str, Enum):
    """
    场景模式。

    STATIC:
        静态场景。任务开始前一次性完成规划，任务过程中不触发重规划。

    DYNAMIC:
        动态场景。任务开始时先做初始规划，任务过程中根据动态事件触发局部或全局重规划。
    """

    STATIC = "static"
    DYNAMIC = "dynamic"


class ReplanningScope(str, Enum):
    """
    动态事件触发后的重规划范围。

    NONE:
        不重规划。

    LOCAL:
        局部重规划。

    AFFECTED_CLUSTERS:
        只重规划受影响的目标群。

    RESOURCE_ONLY:
        只重做资源分配，不重新目标筛选和分群。

    GLOBAL:
        全局重规划，从目标筛选开始重新执行。
    """

    NONE = "none"
    LOCAL = "local"
    AFFECTED_CLUSTERS = "affected_clusters"
    RESOURCE_ONLY = "resource_only"
    GLOBAL = "global"


@dataclass(frozen=True)
class StagePolicy:
    """
    单个阶段的算法配置。

    stage:
        阶段名称，例如 destroy_target_selection、target_clustering。

    method:
        该阶段使用的算法名称，例如 random_forest、pso、rule_based。

    should_run:
        是否需要执行该阶段。
        如果 method 是 keep_existing / skip / none，则 should_run=False。

    reason:
        当前决策的来源说明，便于调试和日志记录。
    """

    stage: str
    method: str
    should_run: bool
    reason: str = ""


@dataclass(frozen=True)
class PolicyDecision:
    """
    算法策略的一次查询结果。

    这个对象告诉上层 MissionPlanner / ReplanningController：
    某个阶段在当前场景、当前事件下应该怎么处理。
    """

    scenario_mode: ScenarioMode
    stage: str
    method: str
    should_run: bool
    replanning_scope: ReplanningScope = ReplanningScope.NONE
    event_type: MissionEventType | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def keep_existing(self) -> bool:
        """是否沿用已有结果。"""
        return self.method == "keep_existing" or not self.should_run


@dataclass(frozen=True)
class EventPolicy:
    """
    某一类动态事件对应的算法策略。

    例如：
        target_appeared:
            replanning_scope: global
            stages:
                target_screening: source_aligned
                destroy_target_selection: random_forest
                target_clustering: pso
                resource_allocation: rule_based
                strike_order_planning: dqn
    """

    event_type: MissionEventType
    replanning_scope: ReplanningScope
    stage_methods: dict[str, str]


@dataclass(frozen=True)
class AlgorithmPolicyConfig:
    """
    算法切换策略配置。

    default_stage_methods:
        默认算法配置。

    static_stage_methods:
        静态场景算法配置。

    dynamic_initial_stage_methods:
        动态场景初始规划算法配置。

    dynamic_event_policies:
        动态事件触发后的算法配置。
    """

    scenario_mode: ScenarioMode = ScenarioMode.STATIC

    default_stage_methods: dict[str, str] = field(default_factory=dict)
    static_stage_methods: dict[str, str] = field(default_factory=dict)
    dynamic_initial_stage_methods: dict[str, str] = field(default_factory=dict)
    dynamic_event_policies: dict[MissionEventType, EventPolicy] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        """检查策略配置是否合法。"""
        if self.scenario_mode not in {ScenarioMode.STATIC, ScenarioMode.DYNAMIC}:
            raise AlgorithmPolicyError(
                f"Unsupported scenario_mode: {self.scenario_mode}"
            )


class AlgorithmPolicy:
    """
    算法切换策略管理器。

    它不直接运行算法，只负责回答：

    1. 当前是静态还是动态场景？
    2. 当前阶段应该用什么算法？
    3. 动态事件发生后，哪些阶段需要重新执行？
    4. 哪些阶段应该 keep_existing？

    这样设计的好处是：
    - 任务流程和算法选择解耦；
    - 后续加入 PPO / XGBoost / ACO / GNN 时，只改配置和对应模块；
    - MissionPlanner 不需要写复杂的 if-else；
    - ReplanningController 可以根据事件类型做局部重规划。
    """

    KEEP_EXISTING_METHODS = {"keep_existing", "skip", "none", "no_op"}

    STAGE_TO_CONFIG_PATH = {
        "destroy_target_selection": "destroy_target_selection.method",
        "target_clustering": "target_clustering.method",
        "resource_allocation": "resource_allocation.method",
        # strike_order_planning 当前还没有统一 method 配置，
        # 先保留映射位置，后续接入 StrikeOrderPlanner 时可以使用。
        "strike_order_planning": "strike_order_planner.method",
    }

    DEFAULT_STAGE_ORDER = [
        "target_screening",
        "destroy_target_selection",
        "target_clustering",
        "resource_allocation",
        "strike_order_planning",
    ]

    def __init__(self, config: AlgorithmPolicyConfig) -> None:
        config.validate()
        self.config = config

    def get_initial_stage_policies(
        self,
        scenario_mode: ScenarioMode | str | None = None,
    ) -> list[PolicyDecision]:
        """
        获取初始规划阶段的算法策略。

        静态场景：
            使用 static_stage_methods。

        动态场景：
            使用 dynamic_initial_stage_methods。

        如果某个阶段没有配置，则回退到 default_stage_methods。
        """
        mode = self._normalize_scenario_mode(
            scenario_mode or self.config.scenario_mode
        )

        if mode == ScenarioMode.STATIC:
            stage_methods = self._merge_stage_methods(
                base=self.config.default_stage_methods,
                override=self.config.static_stage_methods,
            )
            reason = "static_initial_planning"

        else:
            stage_methods = self._merge_stage_methods(
                base=self.config.default_stage_methods,
                override=self.config.dynamic_initial_stage_methods,
            )
            reason = "dynamic_initial_planning"

        decisions: list[PolicyDecision] = []

        for stage in self.DEFAULT_STAGE_ORDER:
            method = stage_methods.get(stage, "keep_existing")
            decisions.append(
                self._build_decision(
                    scenario_mode=mode,
                    stage=stage,
                    method=method,
                    replanning_scope=ReplanningScope.NONE,
                    event_type=None,
                    reason=reason,
                )
            )

        return decisions

    def get_decision(
        self,
        stage: str | PipelineStage,
        scenario_mode: ScenarioMode | str | None = None,
        event_type: MissionEventType | str | None = None,
    ) -> PolicyDecision:
        """
        查询某个阶段在当前条件下应该使用什么算法。

        如果 event_type 为空：
            返回初始规划阶段策略。

        如果 event_type 不为空且 scenario_mode=dynamic：
            返回该事件下的重规划策略。
        """
        normalized_stage = self._normalize_stage(stage)
        mode = self._normalize_scenario_mode(
            scenario_mode or self.config.scenario_mode
        )

        if event_type is None:
            initial_decisions = self.get_initial_stage_policies(mode)

            for decision in initial_decisions:
                if decision.stage == normalized_stage:
                    return decision

            return self._build_decision(
                scenario_mode=mode,
                stage=normalized_stage,
                method="keep_existing",
                replanning_scope=ReplanningScope.NONE,
                event_type=None,
                reason="stage_not_found_in_initial_policy",
            )

        normalized_event = self._normalize_event_type(event_type)

        if mode == ScenarioMode.STATIC:
            return self._build_decision(
                scenario_mode=mode,
                stage=normalized_stage,
                method="keep_existing",
                replanning_scope=ReplanningScope.NONE,
                event_type=normalized_event,
                reason="static_mode_ignores_dynamic_event",
            )

        event_policy = self.config.dynamic_event_policies.get(normalized_event)

        if event_policy is None:
            return self._build_decision(
                scenario_mode=mode,
                stage=normalized_stage,
                method="keep_existing",
                replanning_scope=ReplanningScope.NONE,
                event_type=normalized_event,
                reason="event_policy_not_found",
            )

        method = event_policy.stage_methods.get(normalized_stage, "keep_existing")

        return self._build_decision(
            scenario_mode=mode,
            stage=normalized_stage,
            method=method,
            replanning_scope=event_policy.replanning_scope,
            event_type=normalized_event,
            reason=f"dynamic_event_policy:{normalized_event.value}",
        )

    def get_replanning_decisions(
        self,
        event_type: MissionEventType | str,
        scenario_mode: ScenarioMode | str | None = None,
    ) -> list[PolicyDecision]:
        """
        根据动态事件获取完整重规划策略。

        返回结果按照 DEFAULT_STAGE_ORDER 排序。
        """
        mode = self._normalize_scenario_mode(
            scenario_mode or self.config.scenario_mode
        )
        normalized_event = self._normalize_event_type(event_type)

        decisions: list[PolicyDecision] = []

        for stage in self.DEFAULT_STAGE_ORDER:
            decisions.append(
                self.get_decision(
                    stage=stage,
                    scenario_mode=mode,
                    event_type=normalized_event,
                )
            )

        return decisions

    def build_config_overrides(
        self,
        decisions: list[PolicyDecision],
    ) -> dict[str, Any]:
        """
        根据策略决策生成配置覆盖项。

        例如：
            destroy_target_selection.method = random_forest
            target_clustering.method = pso
            resource_allocation.method = rule_based

        这个函数后续会给 MissionPlanner / ReplanningController 使用。
        """
        overrides: dict[str, Any] = {}

        for decision in decisions:
            if not decision.should_run:
                continue

            config_path = self.STAGE_TO_CONFIG_PATH.get(decision.stage)

            if config_path is None:
                continue

            overrides[config_path] = decision.method

        return overrides

    def summarize_decisions(
        self,
        decisions: list[PolicyDecision],
    ) -> list[dict[str, Any]]:
        """
        将决策列表转换成可打印 / 可写日志的字典。
        """
        return [
            {
                "scenario_mode": decision.scenario_mode.value,
                "stage": decision.stage,
                "method": decision.method,
                "should_run": decision.should_run,
                "keep_existing": decision.keep_existing,
                "replanning_scope": decision.replanning_scope.value,
                "event_type": (
                    decision.event_type.value
                    if decision.event_type is not None
                    else None
                ),
                "reason": decision.reason,
            }
            for decision in decisions
        ]

    def _build_decision(
        self,
        scenario_mode: ScenarioMode,
        stage: str,
        method: str,
        replanning_scope: ReplanningScope,
        event_type: MissionEventType | None,
        reason: str,
    ) -> PolicyDecision:
        """统一构造 PolicyDecision。"""
        normalized_method = str(method)

        should_run = normalized_method not in self.KEEP_EXISTING_METHODS

        return PolicyDecision(
            scenario_mode=scenario_mode,
            stage=stage,
            method=normalized_method,
            should_run=should_run,
            replanning_scope=replanning_scope,
            event_type=event_type,
            reason=reason,
        )

    @staticmethod
    def _merge_stage_methods(
        base: dict[str, str],
        override: dict[str, str],
    ) -> dict[str, str]:
        """合并默认阶段算法和场景专用算法。"""
        merged = dict(base)
        merged.update(override)
        return merged

    @staticmethod
    def _normalize_stage(stage: str | PipelineStage) -> str:
        """
        标准化阶段名称。

        contracts.py 里已有 PipelineStage，但当前工程中也有
        destroy_target_selection 这种更细粒度阶段，所以这里允许字符串。
        """
        if isinstance(stage, PipelineStage):
            return stage.value

        return str(stage)

    @staticmethod
    def _normalize_scenario_mode(
        scenario_mode: ScenarioMode | str,
    ) -> ScenarioMode:
        """标准化场景模式。"""
        if isinstance(scenario_mode, ScenarioMode):
            return scenario_mode

        value = str(scenario_mode).lower()

        try:
            return ScenarioMode(value)
        except ValueError as exc:
            raise AlgorithmPolicyError(
                f"Unsupported scenario_mode: {scenario_mode}"
            ) from exc

    @staticmethod
    def _normalize_event_type(
        event_type: MissionEventType | str,
    ) -> MissionEventType:
        """标准化动态事件类型。"""
        if isinstance(event_type, MissionEventType):
            return event_type

        value = str(event_type).lower()

        try:
            return MissionEventType(value)
        except ValueError as exc:
            raise AlgorithmPolicyError(
                f"Unsupported event_type: {event_type}"
            ) from exc


def load_algorithm_policy_config(
    config: dict[str, Any],
) -> AlgorithmPolicyConfig:
    """
    从项目总配置中读取 AlgorithmPolicyConfig。
    """
    scenario_mode_raw = get_config_value(
        config,
        "scenario.mode",
        default="static",
    )

    scenario_mode = ScenarioMode(str(scenario_mode_raw).lower())

    default_stage_methods = dict(
        get_config_value(
            config,
            "algorithm_policy.default",
            default={},
        )
    )

    static_stage_methods = dict(
        get_config_value(
            config,
            "algorithm_policy.static",
            default={},
        )
    )

    dynamic_initial_stage_methods = dict(
        get_config_value(
            config,
            "algorithm_policy.dynamic.initial_planning",
            default={},
        )
    )

    raw_event_policies = dict(
        get_config_value(
            config,
            "algorithm_policy.dynamic.events",
            default={},
        )
    )

    dynamic_event_policies: dict[MissionEventType, EventPolicy] = {}

    for event_name, raw_event_policy in raw_event_policies.items():
        event_type = AlgorithmPolicy._normalize_event_type(event_name)

        replanning_scope_raw = raw_event_policy.get(
            "replanning_scope",
            "none",
        )
        replanning_scope = ReplanningScope(str(replanning_scope_raw).lower())

        stage_methods = dict(raw_event_policy.get("stages", {}))

        dynamic_event_policies[event_type] = EventPolicy(
            event_type=event_type,
            replanning_scope=replanning_scope,
            stage_methods=stage_methods,
        )

    policy_config = AlgorithmPolicyConfig(
        scenario_mode=scenario_mode,
        default_stage_methods=default_stage_methods,
        static_stage_methods=static_stage_methods,
        dynamic_initial_stage_methods=dynamic_initial_stage_methods,
        dynamic_event_policies=dynamic_event_policies,
    )

    policy_config.validate()
    return policy_config


def load_algorithm_policy(
    config: dict[str, Any],
) -> AlgorithmPolicy:
    """
    从项目总配置中直接构造 AlgorithmPolicy。
    """
    return AlgorithmPolicy(
        config=load_algorithm_policy_config(config)
    )