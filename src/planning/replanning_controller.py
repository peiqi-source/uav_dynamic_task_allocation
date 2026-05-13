"""planning 数据模块中的重规划controller实现。"""
from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from uav_dynamic_task_allocation.allocation.resource_allocation import (
    ResourceAllocator,
    load_resource_allocation_config,
)
from uav_dynamic_task_allocation.allocation.target_clustering import (
    TargetClusterer,
    TargetClusteringResult,
    load_target_clustering_config,
)
from uav_dynamic_task_allocation.core.contracts import (
    AlgorithmMetadata,
    AllocationPlan,
    MissionEvent,
    MissionEventType,
    MissionPlan,
    PipelineStage,
    ResourceStatus,
    ScreenedTargetSet,
    TargetCluster,
    TargetClusterSet,
)
from uav_dynamic_task_allocation.core.entities import Position, Target, UAV
from uav_dynamic_task_allocation.planning.algorithm_policy import (
    AlgorithmPolicy,
    PolicyDecision,
    ReplanningScope,
    ScenarioMode,
)
from uav_dynamic_task_allocation.planning.mission_planner import (
    MissionPlanner,
    MissionPlannerResult,
    load_mission_planner_config,
)
from uav_dynamic_task_allocation.planning.strike_order_planner import (
    StrikeOrderPlannerResult,
    build_strike_order_planner,
)
from uav_dynamic_task_allocation.planning.support_policy import (
    SupportDecision,
    SupportPolicy,
    load_support_policy_config,
)
from uav_dynamic_task_allocation.simulation.mission_state import MissionRuntimeState


class ReplanningControllerError(Exception):
    """动态重规划控制过程中的自定义错误。"""


@dataclass
class RuntimeBattlefieldSnapshot:
    """
    基于 MissionRuntimeState 构造的轻量级战场快照。

    这个对象只提供 MissionPlanner / ResourceAllocator 需要的基本属性：
    - targets
    - uavs
    - attack_uavs
    - guide_uavs
    - communication_uavs

    它不是新的数据模型，只是动态重规划时的适配层。
    """

    # targets: 目标集合。
    targets: list[Target]
    # uavs: uavs 数据。
    uavs: list[UAV]
    # attack_uavs: 攻击uavs。
    attack_uavs: list[UAV]
    # guide_uavs: 导引uavs。
    guide_uavs: list[UAV]
    # communication_uavs: 通信uavs。
    communication_uavs: list[UAV]


@dataclass
class ReplanningResult:
    """
    一次动态重规划结果。

    updated_mission_plan:
        重规划后的任务计划。

    decisions:
        AlgorithmPolicy 对该事件给出的阶段决策。

    replanning_scope:
        本次事件触发的重规划范围。

    metadata:
        保存重规划原因、受影响目标 / UAV、执行了哪些阶段等信息。
    """

    # event: 事件。
    event: MissionEvent
    # previous_mission_plan: previous任务规划方案。
    previous_mission_plan: MissionPlan
    # updated_mission_plan: updated任务规划方案。
    updated_mission_plan: MissionPlan
    # decisions: 决策集合。
    decisions: list[PolicyDecision]
    # replanning_scope: 重规划scope。
    replanning_scope: ReplanningScope
    # mission_planner_result: 任务规划器结果。
    mission_planner_result: MissionPlannerResult | None = None
    # strike_order_planner_result: 打击顺序规划阶段结果。
    strike_order_planner_result: StrikeOrderPlannerResult | None = None
    # support_decision: 支援决策。
    support_decision: SupportDecision | None = None
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)


class ReplanningController:
    """
    动态重规划控制器。

    它负责在动态事件发生后，根据 AlgorithmPolicy 决定如何更新 MissionPlan。

    当前实现的策略：

    1. GLOBAL:
       重新执行 MissionPlanner，从目标筛选开始全流程重规划。

    2. AFFECTED_CLUSTERS / LOCAL:
       当前先做工程化近似：
       - 从已有 ScreenedTargetSet 中移除不可用目标；
       - 从已有 TargetClusterSet 中移除不可用目标；
       - 重新资源分配；
       - 重新群内打击排序。

    3. RESOURCE_ONLY:
       保持目标筛选和目标分群不变；
       根据当前可用 UAV 重新资源分配；
       重新群内打击排序。

    后续可以继续优化成真正的“只重规划受影响 cluster”。
    """

    def __init__(
        self,
        base_config: dict[str, Any],
        algorithm_policy: AlgorithmPolicy,
        logger: logging.Logger | None = None,
    ) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            base_config: 全局基础配置，类型为 dict[str, Any]。
            algorithm_policy: 算法选择策略对象，类型为 AlgorithmPolicy。
            logger: 日志器，类型为 logging.Logger | None。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        # base_config: 全局基础配置。
        self.base_config = base_config
        # algorithm_policy: 算法选择策略对象。
        self.algorithm_policy = algorithm_policy
        # logger: 日志器。
        self.logger = logger or logging.getLogger(__name__)

    def handle_event(
        self,
        runtime_state: MissionRuntimeState,
        event: MissionEvent,
        original_battlefield_state=None,
    ) -> ReplanningResult:
        """
        根据事件执行动态重规划。

        注意：
        调用这个函数前，建议先执行：
            runtime_state.apply_event(event)

        这样 runtime_state 中的可用目标和可用 UAV 已经被更新。
        """
        decisions = self.algorithm_policy.get_replanning_decisions(
            event_type=event.event_type,
            scenario_mode=ScenarioMode.DYNAMIC,
        )

        scope = self._resolve_replanning_scope(decisions)

        self.logger.info(
            "Replanning triggered: "
            f"event={event.event_type.value}, "
            f"time={event.event_time}, "
            f"scope={scope.value}"
        )
        self.logger.info(
            f"Replanning decisions: "
            f"{self.algorithm_policy.summarize_decisions(decisions)}"
        )

        previous_plan = runtime_state.mission_plan

        if scope == ReplanningScope.NONE:
            result = ReplanningResult(
                event=event,
                previous_mission_plan=previous_plan,
                updated_mission_plan=previous_plan,
                decisions=decisions,
                replanning_scope=scope,
                metadata={
                    "action": "keep_existing",
                    "reason": "replanning_scope_none",
                },
            )
            return result

        if scope == ReplanningScope.GLOBAL:
            result = self._run_global_replanning(
                runtime_state=runtime_state,
                event=event,
                decisions=decisions,
                previous_plan=previous_plan,
                original_battlefield_state=original_battlefield_state,
            )

        elif scope in {ReplanningScope.AFFECTED_CLUSTERS, ReplanningScope.LOCAL}:
            result = self._run_filtered_cluster_replanning(
                runtime_state=runtime_state,
                event=event,
                decisions=decisions,
                previous_plan=previous_plan,
                original_battlefield_state=original_battlefield_state,
            )

        elif scope == ReplanningScope.RESOURCE_ONLY:
            result = self._run_resource_only_replanning(
                runtime_state=runtime_state,
                event=event,
                decisions=decisions,
                previous_plan=previous_plan,
                original_battlefield_state=original_battlefield_state,
            )

        else:
            raise ReplanningControllerError(
                f"Unsupported replanning scope: {scope}"
            )

        runtime_state.mission_plan = result.updated_mission_plan

        return result

    def _run_global_replanning(
        self,
        runtime_state: MissionRuntimeState,
        event: MissionEvent,
        decisions: list[PolicyDecision],
        previous_plan: MissionPlan,
        original_battlefield_state,
    ) -> ReplanningResult:
        """
        全局重规划。

        当前用于 target_appeared 这类事件。
        如果新目标对象还没有加载进 runtime_state.active_targets，
        那么全局重规划会基于当前可用目标执行，不会自动生成新目标对象。
        后续可以补 DynamicTargetLoader。
        """
        run_config = self._build_run_config(decisions)

        snapshot = self._build_runtime_snapshot(
            runtime_state=runtime_state,
            original_battlefield_state=original_battlefield_state,
        )

        planner = MissionPlanner(
            base_config=run_config,
            algorithm_policy=self.algorithm_policy,
            planner_config=load_mission_planner_config(run_config),
            logger=self.logger,
        )

        planner_result = planner.plan(
            battlefield_state=snapshot,
            scenario_mode=ScenarioMode.DYNAMIC,
        )

        updated_plan = self._append_replanning_metadata(
            mission_plan=planner_result.mission_plan,
            event=event,
            decisions=decisions,
            scope=ReplanningScope.GLOBAL,
        )
        support_decision = self._apply_support_policy(
            config=run_config,
            event=event,
            allocation_plan=updated_plan.allocation_plan,
        )
        if support_decision is not None:
            updated_plan.metadata.setdefault("support_decisions", []).append(
                support_decision.to_dict()
            )

        return ReplanningResult(
            event=event,
            previous_mission_plan=previous_plan,
            updated_mission_plan=updated_plan,
            decisions=decisions,
            replanning_scope=ReplanningScope.GLOBAL,
            mission_planner_result=planner_result,
            support_decision=support_decision,
            metadata={
                "action": "global_replanning",
                "available_target_ids": runtime_state.available_target_ids,
                "available_uav_ids": runtime_state.available_uav_ids,
                "support_decision": (
                    support_decision.to_dict() if support_decision else None
                ),
            },
        )

    def _run_filtered_cluster_replanning(
        self,
        runtime_state: MissionRuntimeState,
        event: MissionEvent,
        decisions: list[PolicyDecision],
        previous_plan: MissionPlan,
        original_battlefield_state,
    ) -> ReplanningResult:
        """
        目标变化后的局部近似重规划。

        当前先实现成：
        - 过滤不可用目标；
        - 重建 TargetClusterSet；
        - 重做资源分配；
        - 重做打击排序。

        这比全局重规划更保守，也更接近 target_disappeared 的策略。
        """
        run_config = self._build_run_config(decisions)

        available_target_ids = set(runtime_state.available_target_ids)

        filtered_screened_set = self._filter_screened_targets(
            screened_set=previous_plan.screened_targets,
            available_target_ids=available_target_ids,
            event=event,
        )

        target_clustering_result = None
        if self._stage_method(decisions, "target_clustering") not in {
            "keep_existing",
            "skip",
            "none",
            "no_op",
        }:
            target_clustering_result = self._rerun_target_clustering(
                config=run_config,
                screened_targets=filtered_screened_set,
            )
            filtered_cluster_set = target_clustering_result.cluster_set
        else:
            filtered_cluster_set = self._filter_target_clusters(
                cluster_set=previous_plan.target_clusters,
                available_target_ids=available_target_ids,
                event=event,
            )

        resource_status = self._build_resource_status_from_runtime(
            runtime_state=runtime_state,
            original_battlefield_state=original_battlefield_state,
        )

        allocation_plan = self._rerun_resource_allocation(
            config=run_config,
            cluster_set=filtered_cluster_set,
            resource_status=resource_status,
        )
        support_decision = self._apply_support_policy(
            config=run_config,
            event=event,
            allocation_plan=allocation_plan,
        )

        strike_result = self._rerun_strike_order_planning(
            config=run_config,
            cluster_set=filtered_cluster_set,
            allocation_plan=allocation_plan,
        )

        updated_plan = self._build_updated_mission_plan(
            screened_targets=filtered_screened_set,
            target_clusters=filtered_cluster_set,
            allocation_plan=allocation_plan,
            strike_order_plans=strike_result.strike_order_plans,
            previous_plan=previous_plan,
            event=event,
            decisions=decisions,
            scope=ReplanningScope.AFFECTED_CLUSTERS,
            extra_algorithm_metadata=[strike_result.algorithm_metadata],
        )

        return ReplanningResult(
            event=event,
            previous_mission_plan=previous_plan,
            updated_mission_plan=updated_plan,
            decisions=decisions,
            replanning_scope=ReplanningScope.AFFECTED_CLUSTERS,
            strike_order_planner_result=strike_result,
            support_decision=support_decision,
            metadata={
                "action": "filtered_cluster_replanning",
                "available_target_ids": sorted(available_target_ids),
                "num_clusters": filtered_cluster_set.num_clusters,
                "target_clustering_metadata": (
                    target_clustering_result.metadata
                    if target_clustering_result is not None
                    else filtered_cluster_set.metadata
                ),
                "support_decision": (
                    support_decision.to_dict() if support_decision else None
                ),
            },
        )

    def _run_resource_only_replanning(
        self,
        runtime_state: MissionRuntimeState,
        event: MissionEvent,
        decisions: list[PolicyDecision],
        previous_plan: MissionPlan,
        original_battlefield_state,
    ) -> ReplanningResult:
        """
        资源级重规划。

        用于 UAV 损毁 / 资源短缺等事件：
        - 不重新目标筛选；
        - 不重新目标分群；
        - 只根据当前可用 UAV 重做资源分配；
        - 然后重新生成打击排序。
        """
        run_config = self._build_run_config(decisions)

        resource_status = self._build_resource_status_from_runtime(
            runtime_state=runtime_state,
            original_battlefield_state=original_battlefield_state,
        )

        allocation_plan = self._rerun_resource_allocation(
            config=run_config,
            cluster_set=previous_plan.target_clusters,
            resource_status=resource_status,
        )
        support_decision = self._apply_support_policy(
            config=run_config,
            event=event,
            allocation_plan=allocation_plan,
        )

        strike_result = self._rerun_strike_order_planning(
            config=run_config,
            cluster_set=previous_plan.target_clusters,
            allocation_plan=allocation_plan,
        )

        updated_plan = self._build_updated_mission_plan(
            screened_targets=previous_plan.screened_targets,
            target_clusters=previous_plan.target_clusters,
            allocation_plan=allocation_plan,
            strike_order_plans=strike_result.strike_order_plans,
            previous_plan=previous_plan,
            event=event,
            decisions=decisions,
            scope=ReplanningScope.RESOURCE_ONLY,
            extra_algorithm_metadata=[strike_result.algorithm_metadata],
        )

        return ReplanningResult(
            event=event,
            previous_mission_plan=previous_plan,
            updated_mission_plan=updated_plan,
            decisions=decisions,
            replanning_scope=ReplanningScope.RESOURCE_ONLY,
            strike_order_planner_result=strike_result,
            support_decision=support_decision,
            metadata={
                "action": "resource_only_replanning",
                "available_uav_ids": runtime_state.available_uav_ids,
                "damaged_uav_ids": sorted(runtime_state.damaged_uav_ids),
                "support_decision": (
                    support_decision.to_dict() if support_decision else None
                ),
            },
        )

    def _apply_support_policy(
        self,
        config: dict[str, Any],
        event: MissionEvent,
        allocation_plan: AllocationPlan,
    ) -> SupportDecision | None:
        """Run the dynamic support policy after resource allocation."""
        support_config = load_support_policy_config(config)
        if not support_config.enabled:
            return None

        policy = SupportPolicy(support_config)
        decision = policy.decide_and_apply(
            event=event,
            allocation_plan=allocation_plan,
        )
        self.logger.info(f"Support decision: {decision.to_dict()}")
        return decision

    def _rerun_target_clustering(
        self,
        config: dict[str, Any],
        screened_targets: ScreenedTargetSet,
    ) -> TargetClusteringResult:
        """Rerun target clustering during dynamic replanning."""
        clusterer = TargetClusterer(load_target_clustering_config(config))
        result = clusterer.cluster(screened_targets)
        if result.metadata.get("method") == "ppo":
            ppo_metadata = result.metadata.get("ppo_metadata", {})
            self.logger.info(
                "Dynamic PPO target regrouping completed: "
                f"method_used={result.metadata.get('method_used')}, "
                f"fallback_reason={ppo_metadata.get('reason', '')}"
            )
        return result

    def _rerun_resource_allocation(
        self,
        config: dict[str, Any],
        cluster_set: TargetClusterSet,
        resource_status: ResourceStatus,
    ) -> AllocationPlan:
        """重新执行资源分配。"""
        allocator = ResourceAllocator(
            load_resource_allocation_config(config)
        )

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
                "Monte Carlo resource assessment after replanning completed: "
                f"csv={result.allocation_plan.metadata.get('resource_assessment_csv_path')}, "
                f"support_required_cluster_ids={risky_cluster_ids}"
            )

        return result.allocation_plan

    def _rerun_strike_order_planning(
        self,
        config: dict[str, Any],
        cluster_set: TargetClusterSet,
        allocation_plan: AllocationPlan,
    ) -> StrikeOrderPlannerResult:
        """重新执行群内打击排序。"""
        planner = build_strike_order_planner(
            config=config,
            logger=self.logger,
        )

        return planner.plan(
            target_clusters=cluster_set,
            allocation_plan=allocation_plan,
        )

    def _build_runtime_snapshot(
        self,
        runtime_state: MissionRuntimeState,
        original_battlefield_state,
    ) -> RuntimeBattlefieldSnapshot:
        """构造当前可用目标 / UAV 的战场快照。"""
        available_targets = runtime_state.get_available_targets()
        available_uavs = runtime_state.get_available_uavs()

        available_uav_ids = {int(uav.uav_id) for uav in available_uavs}

        if original_battlefield_state is not None:
            original_attack = list(getattr(original_battlefield_state, "attack_uavs", []))
            original_guide = list(getattr(original_battlefield_state, "guide_uavs", []))
            original_comm = list(
                getattr(original_battlefield_state, "communication_uavs", [])
            )

            attack_uavs = [
                uav for uav in original_attack
                if int(uav.uav_id) in available_uav_ids
            ]
            guide_uavs = [
                uav for uav in original_guide
                if int(uav.uav_id) in available_uav_ids
            ]
            communication_uavs = [
                uav for uav in original_comm
                if int(uav.uav_id) in available_uav_ids
            ]
        else:
            attack_uavs = available_uavs
            guide_uavs = []
            communication_uavs = []

        return RuntimeBattlefieldSnapshot(
            targets=available_targets,
            uavs=available_uavs,
            attack_uavs=attack_uavs,
            guide_uavs=guide_uavs,
            communication_uavs=communication_uavs,
        )

    def _build_resource_status_from_runtime(
        self,
        runtime_state: MissionRuntimeState,
        original_battlefield_state,
    ) -> ResourceStatus:
        """根据运行时状态构造当前可用资源池。"""
        snapshot = self._build_runtime_snapshot(
            runtime_state=runtime_state,
            original_battlefield_state=original_battlefield_state,
        )

        damaged_ids = set(runtime_state.damaged_uav_ids)

        damaged_uavs = [
            uav for uav in runtime_state.active_uavs.values()
            if int(uav.uav_id) in damaged_ids
        ]

        return ResourceStatus(
            all_uavs=list(runtime_state.active_uavs.values()),
            available_attack_uavs=snapshot.attack_uavs,
            available_guide_uavs=snapshot.guide_uavs,
            available_communication_uavs=snapshot.communication_uavs,
            damaged_uavs=damaged_uavs,
            metadata={
                "source": "replanning_controller_runtime_resource_status",
                "num_available_attack_uavs": len(snapshot.attack_uavs),
                "num_available_guide_uavs": len(snapshot.guide_uavs),
                "num_available_communication_uavs": len(snapshot.communication_uavs),
                "damaged_uav_ids": sorted(damaged_ids),
            },
        )

    def _filter_screened_targets(
        self,
        screened_set: ScreenedTargetSet,
        available_target_ids: set[int],
        event: MissionEvent,
    ) -> ScreenedTargetSet:
        """过滤 ScreenedTargetSet 中不可用目标。"""
        all_targets = [
            target for target in screened_set.all_targets
            if int(target.target_id) in available_target_ids
        ]

        candidate_targets = [
            target for target in screened_set.candidate_targets
            if int(target.target_id) in available_target_ids
        ]

        selected_targets = [
            target for target in screened_set.selected_targets
            if int(target.target_id) in available_target_ids
        ]

        target_scores = {
            target_id: score
            for target_id, score in screened_set.target_scores.items()
            if int(target_id) in available_target_ids
        }

        filtered = ScreenedTargetSet(
            all_targets=all_targets,
            candidate_targets=candidate_targets,
            selected_targets=selected_targets,
            target_scores=target_scores,
            algorithm_metadata=screened_set.algorithm_metadata,
            metadata={
                **screened_set.metadata,
                "filtered_by_replanning": True,
                "event_type": event.event_type.value,
                "event_time": event.event_time,
                "available_target_ids": sorted(available_target_ids),
            },
        )
        filtered.validate()
        return filtered

    def _filter_target_clusters(
        self,
        cluster_set: TargetClusterSet,
        available_target_ids: set[int],
        event: MissionEvent,
    ) -> TargetClusterSet:
        """过滤 TargetClusterSet 中不可用目标，并重建 cluster 信息。"""
        new_clusters: list[TargetCluster] = []

        for cluster in cluster_set.clusters:
            kept_targets = [
                target for target in cluster.targets
                if int(target.target_id) in available_target_ids
            ]

            if not kept_targets:
                continue

            center = self._calculate_center(kept_targets)

            rebuilt_cluster = TargetCluster(
                cluster_id=cluster.cluster_id,
                targets=kept_targets,
                center=center,
                defense_sum=sum(float(target.defense) for target in kept_targets),
                significance_sum=sum(
                    float(target.significance) for target in kept_targets
                ),
                compactness=self._calculate_compactness(kept_targets, center),
                metadata={
                    **cluster.metadata,
                    "rebuilt_by_replanning": True,
                    "event_type": event.event_type.value,
                    "original_target_ids": cluster.target_ids,
                    "kept_target_ids": [target.target_id for target in kept_targets],
                },
            )
            rebuilt_cluster.validate()
            new_clusters.append(rebuilt_cluster)

        rebuilt_set = TargetClusterSet(
            clusters=new_clusters,
            algorithm_metadata=cluster_set.algorithm_metadata,
            metadata={
                **cluster_set.metadata,
                "rebuilt_by_replanning": True,
                "event_type": event.event_type.value,
                "event_time": event.event_time,
                "num_clusters_after_filter": len(new_clusters),
            },
        )
        rebuilt_set.validate()
        return rebuilt_set

    def _build_updated_mission_plan(
        self,
        screened_targets: ScreenedTargetSet,
        target_clusters: TargetClusterSet,
        allocation_plan: AllocationPlan,
        strike_order_plans: dict,
        previous_plan: MissionPlan,
        event: MissionEvent,
        decisions: list[PolicyDecision],
        scope: ReplanningScope,
        extra_algorithm_metadata: list[AlgorithmMetadata] | None = None,
    ) -> MissionPlan:
        """构造重规划后的 MissionPlan。"""
        mission_stage = getattr(
            PipelineStage,
            "MISSION_PLANNING",
            PipelineStage.RESOURCE_ALLOCATION,
        )

        replanning_metadata = AlgorithmMetadata(
            algorithm_name="replanning_controller",
            algorithm_type="event_driven_replanning",
            stage=mission_stage,
            version="v1",
            config={
                "event_type": event.event_type.value,
                "event_time": event.event_time,
                "replanning_scope": scope.value,
                "stage_methods": {
                    decision.stage: decision.method for decision in decisions
                },
            },
            notes="Event-driven mission replanning result.",
        )

        previous_metadata = list(previous_plan.algorithm_metadata)
        extra_algorithm_metadata = extra_algorithm_metadata or []

        metadata = {
            **previous_plan.metadata,
            "last_replanning_event_type": event.event_type.value,
            "last_replanning_event_time": event.event_time,
            "last_replanning_scope": scope.value,
            "replanned": True,
            "replanning_stage_methods": {
                decision.stage: decision.method for decision in decisions
            },
            "num_clusters": target_clusters.num_clusters,
            "num_assignments": len(allocation_plan.assignments),
            "num_strike_order_plans": len(strike_order_plans),
        }

        updated_plan = MissionPlan(
            screened_targets=screened_targets,
            target_clusters=target_clusters,
            allocation_plan=allocation_plan,
            strike_order_plans=strike_order_plans,
            algorithm_metadata=[
                *previous_metadata,
                replanning_metadata,
                *extra_algorithm_metadata,
            ],
            metadata=metadata,
        )

        updated_plan.validate()
        return updated_plan

    def _append_replanning_metadata(
        self,
        mission_plan: MissionPlan,
        event: MissionEvent,
        decisions: list[PolicyDecision],
        scope: ReplanningScope,
    ) -> MissionPlan:
        """给全局重规划得到的 MissionPlan 追加重规划元信息。"""
        mission_plan.metadata.update(
            {
                "replanned": True,
                "last_replanning_event_type": event.event_type.value,
                "last_replanning_event_time": event.event_time,
                "last_replanning_scope": scope.value,
                "replanning_stage_methods": {
                    decision.stage: decision.method for decision in decisions
                },
            }
        )

        mission_plan.validate()
        return mission_plan

    def _build_run_config(
        self,
        decisions: list[PolicyDecision],
    ) -> dict[str, Any]:
        """根据 AlgorithmPolicy 决策构造本次重规划配置。"""
        run_config = deepcopy(self.base_config)
        overrides = self.algorithm_policy.build_config_overrides(decisions)

        for key_path, value in overrides.items():
            self._set_nested_config_value(run_config, key_path, value)

        return run_config

    def _resolve_replanning_scope(
        self,
        decisions: list[PolicyDecision],
    ) -> ReplanningScope:
        """从决策列表中解析重规划范围。"""
        for decision in decisions:
            if decision.replanning_scope != ReplanningScope.NONE:
                return decision.replanning_scope

        return ReplanningScope.NONE

    @staticmethod
    def _stage_method(
        decisions: list[PolicyDecision],
        stage: str,
    ) -> str:
        """处理阶段method相关业务逻辑。

        参数：
            decisions: 决策集合，类型为 list[PolicyDecision]。
            stage: 阶段，类型为 str。

        返回：
            str，表示该函数计算或构建得到的结果。
        """
        for decision in decisions:
            if decision.stage == stage:
                return decision.method
        return "keep_existing"

    @staticmethod
    def _calculate_center(targets: list[Target]) -> Position:
        """计算目标集合中心。"""
        return Position(
            x=sum(target.position.x for target in targets) / len(targets),
            y=sum(target.position.y for target in targets) / len(targets),
        )

    @staticmethod
    def _calculate_compactness(
        targets: list[Target],
        center: Position,
    ) -> float:
        """计算目标群紧凑度。"""
        if not targets:
            return 0.0

        distances = [
            math.sqrt(
                (target.position.x - center.x) ** 2
                + (target.position.y - center.y) ** 2
            )
            for target in targets
        ]

        return float(sum(distances) / len(distances))

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
