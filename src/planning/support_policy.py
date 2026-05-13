"""planning 数据模块中的支援策略实现。"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from uav_dynamic_task_allocation.core.contracts import (
    AllocationPlan,
    ClusterAssignment,
    MissionEvent,
    MissionEventType,
)
from uav_dynamic_task_allocation.core.entities import UAV
from uav_dynamic_task_allocation.utils.config import get_config_value, resolve_path


class SupportPolicyError(Exception):
    """Raised when dynamic support decisions cannot be produced."""


@dataclass(frozen=True)
class SupportPolicyConfig:
    """Configuration for event-driven UAV support decisions."""

    # enabled: enabled 数据。
    enabled: bool = True
    # completion_threshold: completionthreshold。
    completion_threshold: float = 0.8
    # donor_min_completion_probability: donor最小值completionprobability。
    donor_min_completion_probability: float = 0.85
    # max_reassigned_uavs: 最大值reassigneduavs。
    max_reassigned_uavs: int = 1
    # output_csv_path: 输出CSV 数据路径。
    output_csv_path: str = "outputs/support/support_decisions.csv"

    def validate(self) -> None:
        """校验当前对象或输入配置的合法性。

        参数：
            无显式业务参数。

        返回：
            无返回值；通过状态变更、文件输出或日志记录体现执行结果。
        """
        if self.max_reassigned_uavs < 0:
            raise SupportPolicyError("max_reassigned_uavs must be non-negative.")
        if not 0.0 <= self.completion_threshold <= 1.0:
            raise SupportPolicyError("completion_threshold must be in [0, 1].")


@dataclass
class SupportDecision:
    """Decision record produced by the dynamic support policy."""

    # support_required: 支援required。
    support_required: bool
    # support_type: 支援类型。
    support_type: str
    # event_id: 事件编号。
    event_id: str
    # donor_cluster_id: donor目标簇编号。
    donor_cluster_id: int | None
    # receiver_cluster_id: receiver目标簇编号。
    receiver_cluster_id: int | None
    # reassigned_uav_ids: reassigned无人机编号集合。
    reassigned_uav_ids: list[int] = field(default_factory=list)
    # affected_target_ids: affected目标编号集合。
    affected_target_ids: list[int] = field(default_factory=list)
    # reason: reason 数据。
    reason: str = ""
    # estimated_completion_rate_before: estimatedcompletion率before。
    estimated_completion_rate_before: float | None = None
    # estimated_completion_rate_after: estimatedcompletion率after。
    estimated_completion_rate_after: float | None = None
    # updated_plan: updated规划方案。
    updated_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """将对象转换为字典，便于日志记录、序列化或调试输出。

        参数：
            无显式业务参数。

        返回：
            dict[str, Any]，表示该函数计算或构建得到的结果。
        """
        return asdict(self)


class SupportPolicy:
    """
    Event-driven support policy for fire, reconnaissance and communication aid.

    The policy reads resource-assessment metadata created by the Monte Carlo
    evaluator. When a cluster is below threshold or directly affected by an
    event, it searches for a nearby donor cluster with surplus UAVs and moves a
    small number of UAVs to the receiver assignment.
    """

    def __init__(self, config: SupportPolicyConfig) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            config: 配置对象，类型为 SupportPolicyConfig。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        config.validate()
        # config: 配置。
        self.config = config

    def decide_and_apply(
        self,
        event: MissionEvent,
        allocation_plan: AllocationPlan,
    ) -> SupportDecision:
        """处理decideandapply相关业务逻辑。

        参数：
            event: 事件，类型为 MissionEvent。
            allocation_plan: 资源分配规划方案，类型为 AllocationPlan。

        返回：
            SupportDecision，表示该函数计算或构建得到的结果。
        """
        allocation_plan.validate()

        if not self.config.enabled:
            return self._no_support(event, reason="support_policy_disabled")

        receiver = self._select_receiver(event, allocation_plan.assignments)
        if receiver is None:
            return self._no_support(event, reason="no_receiver_cluster")

        support_type = self._infer_support_type(event, receiver)
        if support_type == "no_support":
            return self._no_support(event, reason="receiver_above_threshold")

        donor = self._select_donor(receiver, allocation_plan.assignments, support_type)
        reassigned = self._reassign_uavs(
            donor=donor,
            receiver=receiver,
            support_type=support_type,
        )

        before = self._completion_probability(receiver)
        after = min(
            1.0,
            before + 0.12 * len(reassigned)
            if reassigned
            else before,
        )

        decision = SupportDecision(
            support_required=True,
            support_type=support_type,
            event_id=self._event_id(event),
            donor_cluster_id=donor.cluster_id if donor is not None else None,
            receiver_cluster_id=receiver.cluster_id,
            reassigned_uav_ids=[uav.uav_id for uav in reassigned],
            affected_target_ids=list(receiver.target_cluster.target_ids),
            reason=self._build_reason(event, receiver, donor, support_type, reassigned),
            estimated_completion_rate_before=before,
            estimated_completion_rate_after=after,
            updated_plan={
                "receiver_attack_uav_ids": receiver.assigned_attack_uav_ids,
                "receiver_guide_uav_ids": receiver.assigned_guide_uav_ids,
                "support_type": support_type,
            },
        )

        allocation_plan.metadata.setdefault("support_decisions", []).append(
            decision.to_dict()
        )
        receiver.metadata.setdefault("support_decisions", []).append(
            decision.to_dict()
        )
        allocation_plan.validate()
        return decision

    def write_csv(
        self,
        decisions: list[SupportDecision],
        output_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> Path:
        """处理writeCSV 数据相关业务逻辑。

        参数：
            decisions: 决策集合，类型为 list[SupportDecision]。
            output_path: 输出路径，类型为 str | Path | None。
            project_root: projectroot，类型为 str | Path | None。

        返回：
            Path，表示该函数计算或构建得到的结果。
        """
        path = resolve_path(
            output_path or self.config.output_csv_path,
            project_root=project_root,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "support_required",
            "support_type",
            "event_id",
            "donor_cluster_id",
            "receiver_cluster_id",
            "reassigned_uav_ids",
            "affected_target_ids",
            "reason",
            "estimated_completion_rate_before",
            "estimated_completion_rate_after",
            "updated_plan",
        ]
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for decision in decisions:
                writer.writerow(decision.to_dict())
        return path

    def _select_receiver(
        self,
        event: MissionEvent,
        assignments: list[ClusterAssignment],
    ) -> ClusterAssignment | None:
        """按照策略从候选集合中选择目标对象，处理receiver 数据相关数据。

        参数：
            event: 事件，类型为 MissionEvent。
            assignments: assignments 数据，类型为 list[ClusterAssignment]。

        返回：
            ClusterAssignment | None，表示该函数计算或构建得到的结果。
        """
        affected_uav_ids = {int(uav_id) for uav_id in event.affected_uav_ids}
        affected_target_ids = {int(target_id) for target_id in event.affected_target_ids}

        for assignment in assignments:
            assignment_uav_ids = {
                *assignment.assigned_attack_uav_ids,
                *assignment.assigned_guide_uav_ids,
                *[uav.uav_id for uav in assignment.assigned_communication_uavs],
            }
            if affected_uav_ids.intersection(assignment_uav_ids):
                return assignment
            if affected_target_ids.intersection(assignment.target_cluster.target_ids):
                return assignment

        risky = [
            assignment for assignment in assignments
            if self._completion_probability(assignment) < self.config.completion_threshold
        ]
        if risky:
            return min(risky, key=self._completion_probability)
        return None

    def _select_donor(
        self,
        receiver: ClusterAssignment,
        assignments: list[ClusterAssignment],
        support_type: str,
    ) -> ClusterAssignment | None:
        """按照策略从候选集合中选择目标对象，处理donor 数据相关数据。

        参数：
            receiver: receiver 数据，类型为 ClusterAssignment。
            assignments: assignments 数据，类型为 list[ClusterAssignment]。
            support_type: 支援类型，类型为 str。

        返回：
            ClusterAssignment | None，表示该函数计算或构建得到的结果。
        """
        candidates: list[ClusterAssignment] = []
        for assignment in assignments:
            if assignment.cluster_id == receiver.cluster_id:
                continue
            if not self._has_surplus(assignment, support_type):
                continue
            if (
                self._completion_probability(assignment)
                < self.config.donor_min_completion_probability
                and not self._has_count_surplus(assignment, support_type)
            ):
                continue
            candidates.append(assignment)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda assignment: assignment.target_cluster.center.distance_to(
                receiver.target_cluster.center
            ),
        )

    def _reassign_uavs(
        self,
        donor: ClusterAssignment | None,
        receiver: ClusterAssignment,
        support_type: str,
    ) -> list[UAV]:
        """处理reassignuavs相关业务逻辑。

        参数：
            donor: donor 数据，类型为 ClusterAssignment | None。
            receiver: receiver 数据，类型为 ClusterAssignment。
            support_type: 支援类型，类型为 str。

        返回：
            list[UAV]，表示该函数计算或构建得到的结果。
        """
        if donor is None or self.config.max_reassigned_uavs == 0:
            return []

        source = self._source_list(donor, support_type)
        target = self._source_list(receiver, support_type)
        count = min(self.config.max_reassigned_uavs, len(source))
        if count <= 0:
            return []

        moved = source[-count:]
        del source[-count:]
        target.extend(moved)

        donor.assigned_attack_uav_count = len(donor.assigned_attack_uavs)
        receiver.assigned_attack_uav_count = len(receiver.assigned_attack_uavs)
        return moved

    @staticmethod
    def _source_list(assignment: ClusterAssignment, support_type: str) -> list[UAV]:
        """处理sourcelist相关业务逻辑。

        参数：
            assignment: assignment 数据，类型为 ClusterAssignment。
            support_type: 支援类型，类型为 str。

        返回：
            list[UAV]，表示该函数计算或构建得到的结果。
        """
        if support_type == "fire_support":
            return assignment.assigned_attack_uavs
        if support_type == "recon_support":
            return assignment.assigned_guide_uavs
        if support_type == "communication_support":
            return assignment.assigned_communication_uavs
        return assignment.assigned_attack_uavs

    def _infer_support_type(
        self,
        event: MissionEvent,
        assignment: ClusterAssignment,
    ) -> str:
        """处理infer支援类型相关业务逻辑。

        参数：
            event: 事件，类型为 MissionEvent。
            assignment: assignment 数据，类型为 ClusterAssignment。

        返回：
            str，表示该函数计算或构建得到的结果。
        """
        if event.event_type == MissionEventType.ATTACK_UAV_DESTROYED:
            return "fire_support"
        if event.event_type == MissionEventType.GUIDE_UAV_DESTROYED:
            return "recon_support"
        if event.event_type == MissionEventType.COMMUNICATION_UAV_DESTROYED:
            return "communication_support"
        if event.event_type in {
            MissionEventType.TARGET_APPEARED,
            MissionEventType.TARGET_DISAPPEARED,
            MissionEventType.RESOURCE_SHORTAGE,
        }:
            assessment = self._assessment(assignment)
            return str(assessment.get("recommended_support_type", "task_reassignment"))
        if self._completion_probability(assignment) < self.config.completion_threshold:
            return "task_reassignment"
        return "no_support"

    def _has_surplus(self, assignment: ClusterAssignment, support_type: str) -> bool:
        """处理hassurplus相关业务逻辑。

        参数：
            assignment: assignment 数据，类型为 ClusterAssignment。
            support_type: 支援类型，类型为 str。

        返回：
            bool，表示该函数计算或构建得到的结果。
        """
        if support_type == "fire_support":
            return len(assignment.assigned_attack_uavs) > 0
        if support_type == "recon_support":
            return len(assignment.assigned_guide_uavs) > 0
        if support_type == "communication_support":
            return len(assignment.assigned_communication_uavs) > 0
        return len(assignment.assigned_attack_uavs) > 0

    def _has_count_surplus(
        self,
        assignment: ClusterAssignment,
        support_type: str,
    ) -> bool:
        """处理hascountsurplus相关业务逻辑。

        参数：
            assignment: assignment 数据，类型为 ClusterAssignment。
            support_type: 支援类型，类型为 str。

        返回：
            bool，表示该函数计算或构建得到的结果。
        """
        if support_type == "fire_support":
            return (
                len(assignment.assigned_attack_uavs)
                > max(assignment.required_attack_uav_count, 1)
            )
        return len(self._source_list(assignment, support_type)) > 1

    @staticmethod
    def _assessment(assignment: ClusterAssignment) -> dict[str, Any]:
        """处理assessment 数据相关业务逻辑。

        参数：
            assignment: assignment 数据，类型为 ClusterAssignment。

        返回：
            dict[str, Any]，表示该函数计算或构建得到的结果。
        """
        return dict(assignment.metadata.get("resource_assessment", {}))

    def _completion_probability(self, assignment: ClusterAssignment) -> float:
        """处理completionprobability相关业务逻辑。

        参数：
            assignment: assignment 数据，类型为 ClusterAssignment。

        返回：
            float，表示该函数计算或构建得到的结果。
        """
        assessment = self._assessment(assignment)
        if "completion_probability" in assessment:
            return float(assessment["completion_probability"])
        if assignment.metadata.get("resource_shortage"):
            return 0.0
        return 1.0

    @staticmethod
    def _event_id(event: MissionEvent) -> str:
        """处理事件编号相关业务逻辑。

        参数：
            event: 事件，类型为 MissionEvent。

        返回：
            str，表示该函数计算或构建得到的结果。
        """
        return f"{event.event_type.value}@{event.event_time:g}"

    def _no_support(self, event: MissionEvent, reason: str) -> SupportDecision:
        """处理no支援相关业务逻辑。

        参数：
            event: 事件，类型为 MissionEvent。
            reason: reason 数据，类型为 str。

        返回：
            SupportDecision，表示该函数计算或构建得到的结果。
        """
        return SupportDecision(
            support_required=False,
            support_type="no_support",
            event_id=self._event_id(event),
            donor_cluster_id=None,
            receiver_cluster_id=None,
            reason=reason,
        )

    @staticmethod
    def _build_reason(
        event: MissionEvent,
        receiver: ClusterAssignment,
        donor: ClusterAssignment | None,
        support_type: str,
        reassigned: list[UAV],
    ) -> str:
        """构建后续流程需要的领域对象或配置对象，处理reason 数据相关数据。

        参数：
            event: 事件，类型为 MissionEvent。
            receiver: receiver 数据，类型为 ClusterAssignment。
            donor: donor 数据，类型为 ClusterAssignment | None。
            support_type: 支援类型，类型为 str。
            reassigned: reassigned 数据，类型为 list[UAV]。

        返回：
            str，表示该函数计算或构建得到的结果。
        """
        if donor is None:
            return (
                f"{support_type} requested by {event.event_type.value}; "
                "no donor cluster with surplus resource was available."
            )
        return (
            f"{support_type} requested by {event.event_type.value}; "
            f"moved {len(reassigned)} UAV(s) from cluster {donor.cluster_id} "
            f"to cluster {receiver.cluster_id}."
        )


def load_support_policy_config(config: dict[str, Any]) -> SupportPolicyConfig:
    """Load support policy configuration from project config."""
    prefix = "support_policy"
    policy_config = SupportPolicyConfig(
        enabled=bool(
            get_config_value(
                config,
                f"{prefix}.enabled",
                default=get_config_value(
                    config,
                    "algorithms.replanning.enable_support_policy",
                    default=True,
                ),
            )
        ),
        completion_threshold=float(
            get_config_value(
                config,
                f"{prefix}.completion_threshold",
                default=get_config_value(
                    config,
                    "monte_carlo.completion_threshold",
                    default=0.8,
                ),
            )
        ),
        donor_min_completion_probability=float(
            get_config_value(
                config,
                f"{prefix}.donor_min_completion_probability",
                default=0.85,
            )
        ),
        max_reassigned_uavs=int(
            get_config_value(
                config,
                f"{prefix}.max_reassigned_uavs",
                default=1,
            )
        ),
        output_csv_path=str(
            get_config_value(
                config,
                f"{prefix}.output_csv_path",
                default="outputs/support/support_decisions.csv",
            )
        ),
    )
    policy_config.validate()
    return policy_config
