"""planning 数据模块中的任务规划器实现。"""
from __future__ import annotations

import csv
import logging
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from uav_dynamic_task_allocation.allocation.resource_allocation import (
    ResourceAllocationResult,
    ResourceAllocator,
    build_resource_status_from_state,
    load_resource_allocation_config,
)
from uav_dynamic_task_allocation.allocation.target_clustering import (
    TargetClusteringResult,
    TargetClusterer,
    load_target_clustering_config,
)
from uav_dynamic_task_allocation.core.contracts import (
    AlgorithmMetadata,
    MissionPlan,
    PipelineStage,
    ScreenedTargetSet,
)
from uav_dynamic_task_allocation.preprocessing.destroy_target_selection import (
    DestroyTargetSelectionResult,
    DestroyTargetSelector,
    load_destroy_target_selection_config,
)
from uav_dynamic_task_allocation.preprocessing.target_screening import (
    TargetScreener,
    load_target_screening_config,
)
from uav_dynamic_task_allocation.planning.algorithm_policy import (
    AlgorithmPolicy,
    PolicyDecision,
    ScenarioMode,
)
from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    resolve_path,
)
from uav_dynamic_task_allocation.planning.strike_order_planner import (
    StrikeOrderPlannerResult,
    build_strike_order_planner,
)


class MissionPlannerError(Exception):
    """任务规划流程中的自定义错误。"""


@dataclass(frozen=True)
class MissionPlannerConfig:
    """
    MissionPlanner 配置。

    MissionPlanner 不直接实现具体算法，而是根据 AlgorithmPolicy 的决策，
    调用目标评分、摧毁目标集选择、目标分群和资源分配模块。
    """

    # debug_csv_path: 调试 CSV 输出路径。
    debug_csv_path: str = "outputs/intermediate/mission_planner_summary.csv"

    def validate(self) -> None:
        """检查配置是否合法。"""
        if not self.debug_csv_path:
            raise MissionPlannerError("debug_csv_path must not be empty.")


@dataclass
class MissionPlannerResult:
    """
    一次任务规划结果。

    mission_plan:
        标准化 MissionPlan，后续 MissionSimulator 会直接使用。

    decisions:
        AlgorithmPolicy 给出的阶段算法决策。

    stage_outputs:
        保存每个阶段的中间输出，便于调试和后续动态重规划复用。
    """

    # mission_plan: 标准任务规划方案。
    mission_plan: MissionPlan
    # decisions: 决策集合。
    decisions: list[PolicyDecision]

    # screened_set: 目标筛选阶段输出集合。
    screened_set: ScreenedTargetSet
    # selected_screened_set: 进入后续规划的筛选目标集合。
    selected_screened_set: ScreenedTargetSet

    # destroy_selection_result: 摧毁目标选择阶段结果。
    destroy_selection_result: DestroyTargetSelectionResult
    # target_clustering_result: 目标分群阶段结果。
    target_clustering_result: TargetClusteringResult
    # resource_allocation_result: 资源分配阶段结果。
    resource_allocation_result: ResourceAllocationResult
    # strike_order_planner_result: 打击顺序规划阶段结果。
    strike_order_planner_result: StrikeOrderPlannerResult

    # stage_outputs: 各阶段中间输出字典。
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)


class MissionPlanner:
    """
    任务规划器。

    该类负责把当前已经完成的模块串成完整规划流程：

        BattlefieldState
            ↓
        TargetScreener
            ↓
        DestroyTargetSelector
            ↓
        TargetClusterer
            ↓
        ResourceAllocator
            ↓
        MissionPlan

    它支持静态场景和动态场景的初始规划。
    动态事件后的局部重规划由后续 ReplanningController 负责。
    """

    def __init__(
        self,
        base_config: dict[str, Any],
        algorithm_policy: AlgorithmPolicy,
        planner_config: MissionPlannerConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            base_config: 全局基础配置，类型为 dict[str, Any]。
            algorithm_policy: 算法选择策略对象，类型为 AlgorithmPolicy。
            planner_config: 任务规划器配置，类型为 MissionPlannerConfig。
            logger: 日志器，类型为 logging.Logger | None。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        planner_config.validate()

        # base_config: 全局基础配置。
        self.base_config = base_config
        # algorithm_policy: 算法选择策略对象。
        self.algorithm_policy = algorithm_policy
        # config: 配置。
        self.config = planner_config
        # logger: 日志器。
        self.logger = logger or logging.getLogger(__name__)

    def plan(
        self,
        battlefield_state,
        scenario_mode: ScenarioMode | str | None = None,
    ) -> MissionPlannerResult:
        """
        执行一次完整任务规划。

        Args:
            battlefield_state:
                由 build_battlefield_state 构造的战场状态。

            scenario_mode:
                static / dynamic。
                如果不传，则使用 AlgorithmPolicyConfig 中的 scenario_mode。

        Returns:
            MissionPlannerResult。
        """
        decisions = self.algorithm_policy.get_initial_stage_policies(
            scenario_mode=scenario_mode
        )

        run_config = self._build_run_config(decisions)

        self.logger.info("Mission planning started.")
        self.logger.info(
            f"Policy decisions: {self.algorithm_policy.summarize_decisions(decisions)}"
        )
        self.logger.info(
            f"Config overrides: {self.algorithm_policy.build_config_overrides(decisions)}"
        )

        # 1. 目标评分
        screened_set = self._run_target_screening(
            config=run_config,
            battlefield_state=battlefield_state,
        )

        # 2. 摧毁目标集选择
        destroy_selection_result = self._run_destroy_target_selection(
            config=run_config,
            screened_set=screened_set,
        )
        selected_screened_set = destroy_selection_result.to_screened_target_set()

        # 3. 目标分群
        target_clustering_result = self._run_target_clustering(
            config=run_config,
            selected_screened_set=selected_screened_set,
        )

        # 4. UAV 资源池
        resource_status = build_resource_status_from_state(battlefield_state)

        # 5. UAV-目标群资源分配
        resource_allocation_result = self._run_resource_allocation(
            config=run_config,
            cluster_set=target_clustering_result.cluster_set,
            resource_status=resource_status,
        )

        # 6. 群内打击次序规划
        strike_order_planner_result = self._run_strike_order_planning(
            config=run_config,
            target_clusters=target_clustering_result.cluster_set,
            allocation_plan=resource_allocation_result.allocation_plan,
        )

        # 7. 构造 MissionPlan
        mission_plan = self._build_mission_plan(
            screened_set=selected_screened_set,
            target_clustering_result=target_clustering_result,
            resource_allocation_result=resource_allocation_result,
            strike_order_planner_result=strike_order_planner_result,
            decisions=decisions,
        )

        metadata = {
            "scenario_mode": (
                decisions[0].scenario_mode.value if decisions else None
            ),
            "num_all_targets": len(screened_set.all_targets),
            "num_candidate_targets": len(screened_set.candidate_targets),
            "num_destroy_targets": len(selected_screened_set.selected_targets),
            "num_target_clusters": target_clustering_result.cluster_set.num_clusters,
            "num_assignments": len(
                resource_allocation_result.allocation_plan.assignments
            ),
            "num_strike_order_plans": len(
                strike_order_planner_result.strike_order_plans
            ),
            "strike_order_method_used": strike_order_planner_result.metadata.get(
                "method_used"
            ),
            "stage_methods": {
                decision.stage: decision.method for decision in decisions
            },
        }

        result = MissionPlannerResult(
            mission_plan=mission_plan,
            decisions=decisions,
            screened_set=screened_set,
            selected_screened_set=selected_screened_set,
            destroy_selection_result=destroy_selection_result,
            target_clustering_result=target_clustering_result,
            resource_allocation_result=resource_allocation_result,
            strike_order_planner_result=strike_order_planner_result,
            stage_outputs={
                "screened_set": screened_set,
                "selected_screened_set": selected_screened_set,
                "destroy_selection_result": destroy_selection_result,
                "target_clustering_result": target_clustering_result,
                "resource_allocation_result": resource_allocation_result,
                "strike_order_planner_result": strike_order_planner_result,
            },
            metadata=metadata,
        )

        self.logger.info(f"Mission planning finished successfully: {metadata}")

        return result

    def write_debug_csv(
        self,
        result: MissionPlannerResult,
        output_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> Path:
        """
        保存 MissionPlanner 阶段摘要。
        """
        path = resolve_path(
            output_path or self.config.debug_csv_path,
            project_root=project_root,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "stage",
            "method",
            "should_run",
            "keep_existing",
            "scenario_mode",
            "replanning_scope",
            "event_type",
            "reason",
        ]

        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for decision in result.decisions:
                writer.writerow(
                    {
                        "stage": decision.stage,
                        "method": decision.method,
                        "should_run": decision.should_run,
                        "keep_existing": decision.keep_existing,
                        "scenario_mode": decision.scenario_mode.value,
                        "replanning_scope": decision.replanning_scope.value,
                        "event_type": (
                            decision.event_type.value
                            if decision.event_type is not None
                            else ""
                        ),
                        "reason": decision.reason,
                    }
                )

        return path

    def _build_run_config(
        self,
        decisions: list[PolicyDecision],
    ) -> dict[str, Any]:
        """
        根据 AlgorithmPolicy 决策生成本次规划使用的配置。

        注意：
            不修改原始 base_config，只在 deepcopy 后的配置上应用 overrides。
        """
        run_config = deepcopy(self.base_config)
        overrides = self.algorithm_policy.build_config_overrides(decisions)

        for key_path, value in overrides.items():
            self._set_nested_config_value(run_config, key_path, value)

        return run_config

    def _run_target_screening(
        self,
        config: dict[str, Any],
        battlefield_state,
    ) -> ScreenedTargetSet:
        """运行目标评分阶段。"""
        screening_config = load_target_screening_config(config)
        screener = TargetScreener(screening_config)

        screened_set = screener.screen(battlefield_state.targets)
        screened_set.validate()

        return screened_set

    def _run_destroy_target_selection(
        self,
        config: dict[str, Any],
        screened_set: ScreenedTargetSet,
    ) -> DestroyTargetSelectionResult:
        """运行摧毁目标集选择阶段。"""
        selector_config = load_destroy_target_selection_config(config)
        selector = DestroyTargetSelector(selector_config)

        result = selector.select(screened_set)
        return result

    def _run_target_clustering(
        self,
        config: dict[str, Any],
        selected_screened_set: ScreenedTargetSet,
    ) -> TargetClusteringResult:
        """运行目标分群阶段。"""
        clustering_config = load_target_clustering_config(config)
        clusterer = TargetClusterer(clustering_config)

        result = clusterer.cluster(selected_screened_set)
        if result.metadata.get("method") == "ppo":
            ppo_metadata = result.metadata.get("ppo_metadata", {})
            self.logger.info(
                "PPO target regrouping completed: "
                f"method_used={result.metadata.get('method_used')}, "
                f"fallback_reason={ppo_metadata.get('reason', '')}"
            )
        return result

    def _run_resource_allocation(
        self,
        config: dict[str, Any],
        cluster_set,
        resource_status,
    ) -> ResourceAllocationResult:
        """运行 UAV-目标群资源分配阶段。"""
        allocation_config = load_resource_allocation_config(config)
        allocator = ResourceAllocator(allocation_config)

        result = allocator.allocate(
            cluster_set=cluster_set,
            resource_status=resource_status,
        )
        assessment = result.allocation_plan.metadata.get("resource_assessment", [])
        if assessment:
            risky_cluster_ids = [
                item.get("cluster_id")
                for item in assessment
                if item.get("support_required")
            ]
            self.logger.info(
                "Monte Carlo resource assessment completed: "
                f"csv={result.allocation_plan.metadata.get('resource_assessment_csv_path')}, "
                f"figure={result.allocation_plan.metadata.get('resource_assessment_figure_path')}, "
                f"support_required_cluster_ids={risky_cluster_ids}"
            )
        return result

    def _run_strike_order_planning(
            self,
            config: dict[str, Any],
            target_clusters,
            allocation_plan,
    ) -> StrikeOrderPlannerResult:
        """
        运行群内打击次序规划阶段。

        该阶段会为每个 TargetCluster 生成 StrikeOrderPlan。

        当前支持：
        - dqn
        - nearest_neighbor
        - random

        如果配置 method=dqn 但 checkpoint 不存在，并且 allow_fallback=True，
        会自动退回到 fallback_method，例如 nearest_neighbor。
        """
        strike_order_planner = build_strike_order_planner(
            config=config,
            logger=self.logger,
        )

        result = strike_order_planner.plan(
            target_clusters=target_clusters,
            allocation_plan=allocation_plan,
        )

        return result

    def _build_mission_plan(
            self,
            screened_set: ScreenedTargetSet,
            target_clustering_result: TargetClusteringResult,
            resource_allocation_result: ResourceAllocationResult,
            strike_order_planner_result: StrikeOrderPlannerResult,
            decisions: list[PolicyDecision],
    ) -> MissionPlan:
        """
        构造 MissionPlan。

        这里严格对齐 core/contracts.py 中 MissionPlan 的字段定义。
        不使用多字段名自动适配，避免隐藏数据合同不统一的问题。
        """

        mission_stage = getattr(
            PipelineStage,
            "MISSION_PLANNING",
            PipelineStage.RESOURCE_ALLOCATION,
        )

        algorithm_metadata = AlgorithmMetadata(
            algorithm_name="mission_planner",
            algorithm_type="pipeline",
            stage=mission_stage,
            version="v1",
            config={
                "stage_methods": {
                    decision.stage: decision.method for decision in decisions
                }
            },
            notes=(
                "Mission planning pipeline. "
                "It integrates target screening, destroy target selection, "
                "target clustering, and resource allocation."
            ),
        )

        metadata = {
            "stage_methods": {
                decision.stage: decision.method for decision in decisions
            },
            "num_destroy_targets": len(screened_set.selected_targets),
            "num_clusters": target_clustering_result.cluster_set.num_clusters,
            "num_assignments": len(
                resource_allocation_result.allocation_plan.assignments
            ),
            "strike_order_status": "integrated",
            "num_strike_order_plans": len(
                strike_order_planner_result.strike_order_plans
            ),
            "strike_order_method_used": strike_order_planner_result.metadata.get(
                "method_used"
            ),
        }

        algorithm_metadata_list = [
            item for item in [
                algorithm_metadata,
                screened_set.algorithm_metadata,
                target_clustering_result.cluster_set.algorithm_metadata,
                resource_allocation_result.allocation_plan.algorithm_metadata,
                strike_order_planner_result.algorithm_metadata,
            ]
            if item is not None
        ]

        mission_plan = MissionPlan(
            screened_targets=screened_set,
            target_clusters=target_clustering_result.cluster_set,
            allocation_plan=resource_allocation_result.allocation_plan,
            strike_order_plans=strike_order_planner_result.strike_order_plans,
            algorithm_metadata=algorithm_metadata_list,
            metadata=metadata,
        )

        mission_plan.validate()

        return mission_plan

    @staticmethod
    def _set_nested_config_value(
        config: dict[str, Any],
        key_path: str,
        value: Any,
    ) -> None:
        """修改嵌套配置字典。"""
        keys = key_path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value


def load_mission_planner_config(
    config: dict[str, Any],
) -> MissionPlannerConfig:
    """从项目总配置中读取 MissionPlannerConfig。"""
    planner_config = MissionPlannerConfig(
        debug_csv_path=str(
            get_config_value(
                config,
                "mission_planner.output.debug_csv_path",
                default="outputs/intermediate/mission_planner_summary.csv",
            )
        )
    )

    planner_config.validate()
    return planner_config
