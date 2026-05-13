from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from uav_dynamic_task_allocation.allocation.monte_carlo_resource_evaluator import (
    MonteCarloResourceEvaluator,
    MonteCarloResourceEvaluatorConfig,
)
from uav_dynamic_task_allocation.core.contracts import (
    AlgorithmMetadata,
    AllocationPlan,
    ClusterAssignment,
    PipelineStage,
    ResourceStatus,
    TargetCluster,
    TargetClusterSet,
)
from uav_dynamic_task_allocation.core.entities import Position, UAV
from uav_dynamic_task_allocation.utils.config import get_config_value, resolve_path


class ResourceAllocationError(Exception):
    """UAV 资源分配过程中的自定义错误。"""


@dataclass(frozen=True)
class ResourceAllocationConfig:
    """
    UAV 资源分配配置。

    当前版本先实现 rule_based 方法，对齐源代码 main.py 中的基本思想：
    根据目标群总防御值 defense_sum 和单架攻击 UAV 的载荷能力，
    估计每个目标群需要多少攻击 UAV。

    后续可以扩展：
    - greedy allocation
    - auction-based allocation
    - Hungarian matching
    - PSO allocation
    - PPO allocation

    但所有方法都应该输出 AllocationPlan，
    这样后续 StrikeOrderDQN 和 MissionSimulator 不需要改。
    """

    method: str = "rule_based"

    attack_payload_per_uav: float = 6.0
    guide_uavs_per_cluster: int = 1
    communication_uavs_per_cluster: int = 0

    allow_partial_allocation: bool = True
    enable_monte_carlo_assessment: bool = True
    monte_carlo_config: MonteCarloResourceEvaluatorConfig = field(
        default_factory=MonteCarloResourceEvaluatorConfig
    )

    cluster_priority: str = "defense_desc"
    uav_sorting: str = "uav_id_asc"

    base_position: Position = field(
        default_factory=lambda: Position(x=0.0, y=-16000.0)
    )

    debug_csv_path: str = "outputs/intermediate/resource_allocation.csv"

    def validate(self) -> None:
        """检查配置是否合法。"""
        supported_methods = {
            "rule_based",
        }

        if self.method not in supported_methods:
            raise ResourceAllocationError(
                f"Unsupported resource allocation method={self.method}. "
                f"Supported methods: {sorted(supported_methods)}"
            )

        if self.attack_payload_per_uav <= 0:
            raise ResourceAllocationError("attack_payload_per_uav must be positive.")

        if self.guide_uavs_per_cluster < 0:
            raise ResourceAllocationError("guide_uavs_per_cluster must be >= 0.")

        if self.communication_uavs_per_cluster < 0:
            raise ResourceAllocationError(
                "communication_uavs_per_cluster must be >= 0."
            )
        self.monte_carlo_config.validate()

        supported_cluster_priority = {
            "defense_desc",
            "significance_desc",
            "num_targets_desc",
            "cluster_id_asc",
        }

        if self.cluster_priority not in supported_cluster_priority:
            raise ResourceAllocationError(
                f"Unsupported cluster_priority={self.cluster_priority}. "
                f"Supported: {sorted(supported_cluster_priority)}"
            )

        supported_uav_sorting = {
            "uav_id_asc",
            "uav_id_desc",
        }

        if self.uav_sorting not in supported_uav_sorting:
            raise ResourceAllocationError(
                f"Unsupported uav_sorting={self.uav_sorting}. "
                f"Supported: {sorted(supported_uav_sorting)}"
            )


@dataclass
class ResourceAllocationRecord:
    """
    单个目标群资源分配记录。

    用于 debug CSV 和后续分析。
    """

    cluster_id: int
    target_ids: list[int]
    required_attack_uav_count: int
    assigned_attack_uav_ids: list[int]
    assigned_guide_uav_ids: list[int]
    assigned_communication_uav_ids: list[int]
    defense_sum: float
    significance_sum: float
    cluster_center: Position
    resource_shortage: bool
    metadata: dict[str, Any]


@dataclass
class ResourceAllocationResult:
    """
    资源分配结果。

    allocation_plan:
        标准化输出，后续 MissionSimulator 和 StrikeOrder planner 会使用。

    records:
        每个目标群的资源分配细节，便于保存和检查。

    metadata:
        保存资源剩余情况、短缺情况等。
    """

    allocation_plan: AllocationPlan
    records: list[ResourceAllocationRecord]
    metadata: dict[str, Any]


class ResourceAllocator:
    """
    UAV 资源分配器。

    输入：
        TargetClusterSet
        ResourceStatus

    输出：
        AllocationPlan

    当前阶段做的是目标群级资源匹配：
        TargetCluster → UAV group

    不是单架 UAV 到单个 target 的最终打击动作。
    群内目标打击顺序由后续 StrikeOrder DQN 负责。
    """

    def __init__(self, config: ResourceAllocationConfig) -> None:
        config.validate()
        self.config = config

    def allocate(
        self,
        cluster_set: TargetClusterSet,
        resource_status: ResourceStatus,
    ) -> ResourceAllocationResult:
        """
        对目标群进行 UAV 资源分配。
        """
        cluster_set.validate()

        if not cluster_set.clusters:
            raise ResourceAllocationError("cluster_set.clusters must not be empty.")

        if not resource_status.available_attack_uavs:
            if not self.config.allow_partial_allocation:
                raise ResourceAllocationError(
                    "No available attack UAVs, and allow_partial_allocation=False."
                )

        if self.config.method == "rule_based":
            return self._allocate_by_rule_based(
                cluster_set=cluster_set,
                resource_status=resource_status,
            )

        raise ResourceAllocationError(
            f"Unsupported resource allocation method: {self.config.method}"
        )

    def write_debug_csv(
        self,
        result: ResourceAllocationResult,
        output_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> Path:
        """
        保存资源分配结果，便于检查和报告分析。
        """
        path = resolve_path(
            output_path or self.config.debug_csv_path,
            project_root=project_root,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "cluster_id",
            "target_ids",
            "num_targets",
            "defense_sum",
            "significance_sum",
            "cluster_center_x",
            "cluster_center_y",
            "required_attack_uav_count",
            "assigned_attack_uav_count",
            "assigned_attack_uav_ids",
            "assigned_guide_uav_ids",
            "assigned_communication_uav_ids",
            "resource_shortage",
            "metadata",
        ]

        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for record in result.records:
                writer.writerow(
                    {
                        "cluster_id": record.cluster_id,
                        "target_ids": record.target_ids,
                        "num_targets": len(record.target_ids),
                        "defense_sum": record.defense_sum,
                        "significance_sum": record.significance_sum,
                        "cluster_center_x": record.cluster_center.x,
                        "cluster_center_y": record.cluster_center.y,
                        "required_attack_uav_count": record.required_attack_uav_count,
                        "assigned_attack_uav_count": len(
                            record.assigned_attack_uav_ids
                        ),
                        "assigned_attack_uav_ids": record.assigned_attack_uav_ids,
                        "assigned_guide_uav_ids": record.assigned_guide_uav_ids,
                        "assigned_communication_uav_ids": (
                            record.assigned_communication_uav_ids
                        ),
                        "resource_shortage": record.resource_shortage,
                        "metadata": record.metadata,
                    }
                )

        return path

    def _allocate_by_rule_based(
        self,
        cluster_set: TargetClusterSet,
        resource_status: ResourceStatus,
    ) -> ResourceAllocationResult:
        """
        规则式 UAV-目标群资源分配。

        基本逻辑：
        1. 根据 cluster_priority 对目标群排序；
        2. 根据 defense_sum / attack_payload_per_uav 计算所需攻击 UAV；
        3. 依次分配攻击 UAV；
        4. 每个目标群分配固定数量导引 UAV / 通信 UAV；
        5. 如果资源不足，根据 allow_partial_allocation 决定是否继续。
        """
        available_attack_uavs = self._sort_uavs(
            list(resource_status.available_attack_uavs)
        )
        available_guide_uavs = self._sort_uavs(
            list(resource_status.available_guide_uavs)
        )
        available_communication_uavs = self._sort_uavs(
            list(resource_status.available_communication_uavs)
        )

        sorted_clusters = self._sort_clusters(cluster_set.clusters)

        assignments: list[ClusterAssignment] = []
        records: list[ResourceAllocationRecord] = []

        total_required_attack_uavs = 0
        total_assigned_attack_uavs = 0
        shortage_cluster_ids: list[int] = []

        for cluster in sorted_clusters:
            required_attack_count = self._estimate_required_attack_uavs(cluster)
            total_required_attack_uavs += required_attack_count

            assigned_attack_uavs = self._take_uavs(
                available_attack_uavs,
                required_attack_count,
            )

            resource_shortage = len(assigned_attack_uavs) < required_attack_count

            if resource_shortage:
                shortage_cluster_ids.append(cluster.cluster_id)

                if not self.config.allow_partial_allocation:
                    raise ResourceAllocationError(
                        "Insufficient attack UAVs for cluster "
                        f"{cluster.cluster_id}: "
                        f"required={required_attack_count}, "
                        f"available={len(assigned_attack_uavs)}"
                    )

            assigned_guide_uavs = self._take_uavs(
                available_guide_uavs,
                self.config.guide_uavs_per_cluster,
            )

            assigned_communication_uavs = self._take_uavs(
                available_communication_uavs,
                self.config.communication_uavs_per_cluster,
            )

            total_assigned_attack_uavs += len(assigned_attack_uavs)

            assignment = ClusterAssignment(
                cluster_id=cluster.cluster_id,
                target_cluster=cluster,
                assigned_attack_uavs=assigned_attack_uavs,
                assigned_guide_uavs=assigned_guide_uavs,
                assigned_communication_uavs=assigned_communication_uavs,
                required_attack_uav_count=required_attack_count,
                assigned_attack_uav_count=len(assigned_attack_uavs),
                start_position=self.config.base_position,
                real_time_position=self.config.base_position,
                target_center=cluster.center,
                metadata={
                    "method": "rule_based",
                    "resource_shortage": resource_shortage,
                    "cluster_priority": self.config.cluster_priority,
                    "attack_payload_per_uav": self.config.attack_payload_per_uav,
                },
            )
            assignment.validate()
            assignments.append(assignment)

            record = ResourceAllocationRecord(
                cluster_id=cluster.cluster_id,
                target_ids=cluster.target_ids,
                required_attack_uav_count=required_attack_count,
                assigned_attack_uav_ids=assignment.assigned_attack_uav_ids,
                assigned_guide_uav_ids=assignment.assigned_guide_uav_ids,
                assigned_communication_uav_ids=[
                    uav.uav_id for uav in assigned_communication_uavs
                ],
                defense_sum=cluster.defense_sum,
                significance_sum=cluster.significance_sum,
                cluster_center=cluster.center,
                resource_shortage=resource_shortage,
                metadata={
                    "num_targets": cluster.num_targets,
                    "compactness": cluster.compactness,
                },
            )
            records.append(record)

        remaining_resource_status = ResourceStatus(
            all_uavs=resource_status.all_uavs,
            available_attack_uavs=available_attack_uavs,
            available_guide_uavs=available_guide_uavs,
            available_communication_uavs=available_communication_uavs,
            damaged_uavs=resource_status.damaged_uavs,
            metadata={
                "source": "resource_allocation_remaining_status",
                "method": self.config.method,
            },
        )

        allocation_plan = AllocationPlan(
            assignments=assignments,
            resource_status=remaining_resource_status,
            algorithm_metadata=AlgorithmMetadata(
                algorithm_name="rule_based_resource_allocator",
                algorithm_type="rule_based",
                stage=PipelineStage.RESOURCE_ALLOCATION,
                version="v1",
                config={
                    "attack_payload_per_uav": self.config.attack_payload_per_uav,
                    "guide_uavs_per_cluster": self.config.guide_uavs_per_cluster,
                    "communication_uavs_per_cluster": (
                        self.config.communication_uavs_per_cluster
                    ),
                    "allow_partial_allocation": self.config.allow_partial_allocation,
                    "cluster_priority": self.config.cluster_priority,
                    "uav_sorting": self.config.uav_sorting,
                    "base_position": [
                        self.config.base_position.x,
                        self.config.base_position.y,
                    ],
                },
                notes=(
                    "Rule-based resource allocation. "
                    "It assigns attack UAVs according to cluster defense_sum "
                    "and attack payload capacity."
                ),
            ),
            metadata={
                "method": "rule_based",
                "total_required_attack_uavs": total_required_attack_uavs,
                "total_assigned_attack_uavs": total_assigned_attack_uavs,
                "shortage_cluster_ids": shortage_cluster_ids,
                "remaining_attack_uavs": [
                    uav.uav_id for uav in available_attack_uavs
                ],
                "remaining_guide_uavs": [
                    uav.uav_id for uav in available_guide_uavs
                ],
                "remaining_communication_uavs": [
                    uav.uav_id for uav in available_communication_uavs
                ],
            },
        )
        allocation_plan.validate()

        assessment_results = []
        assessment_csv_path = None
        assessment_figure_path = None
        if self.config.enable_monte_carlo_assessment:
            evaluator = MonteCarloResourceEvaluator(self.config.monte_carlo_config)
            assessment_results = evaluator.assess_allocation(allocation_plan)
            assessment_csv_path = evaluator.write_csv(assessment_results)
            assessment_samples_csv_path = evaluator.write_samples_csv(
                assessment_results
            )
            assessment_figure_path = evaluator.write_distribution_plot(
                assessment_results
            )
            allocation_plan.metadata["resource_assessment"] = [
                result.to_dict() for result in assessment_results
            ]
            allocation_plan.metadata["resource_assessment_csv_path"] = str(
                assessment_csv_path
            )
            allocation_plan.metadata["resource_assessment_samples_csv_path"] = str(
                assessment_samples_csv_path
            )
            allocation_plan.metadata["resource_assessment_figure_path"] = (
                str(assessment_figure_path) if assessment_figure_path else None
            )
            for assignment in allocation_plan.assignments:
                matching_result = next(
                    (
                        result for result in assessment_results
                        if result.cluster_id == assignment.cluster_id
                    ),
                    None,
                )
                if matching_result is not None:
                    assignment.metadata["resource_assessment"] = (
                        matching_result.to_dict()
                    )

        return ResourceAllocationResult(
            allocation_plan=allocation_plan,
            records=records,
            metadata=allocation_plan.metadata,
        )

    def _estimate_required_attack_uavs(
        self,
        cluster: TargetCluster,
    ) -> int:
        """
        根据目标群防御总值估计攻击 UAV 需求。

        对齐源代码 main.py 中类似：
            required_attack_uavs = ceil(cluster_defense_sum / attack_payload_per_uav)

        同时至少分配 1 架攻击 UAV，避免目标群没有打击资源。
        """
        required = math.ceil(
            float(cluster.defense_sum) / self.config.attack_payload_per_uav
        )

        return max(1, int(required))

    def _sort_clusters(
        self,
        clusters: list[TargetCluster],
    ) -> list[TargetCluster]:
        """根据配置对目标群排序。"""
        if self.config.cluster_priority == "defense_desc":
            return sorted(
                clusters,
                key=lambda cluster: cluster.defense_sum,
                reverse=True,
            )

        if self.config.cluster_priority == "significance_desc":
            return sorted(
                clusters,
                key=lambda cluster: cluster.significance_sum,
                reverse=True,
            )

        if self.config.cluster_priority == "num_targets_desc":
            return sorted(
                clusters,
                key=lambda cluster: cluster.num_targets,
                reverse=True,
            )

        if self.config.cluster_priority == "cluster_id_asc":
            return sorted(
                clusters,
                key=lambda cluster: cluster.cluster_id,
            )

        raise ResourceAllocationError(
            f"Unsupported cluster_priority: {self.config.cluster_priority}"
        )

    def _sort_uavs(
        self,
        uavs: list[UAV],
    ) -> list[UAV]:
        """根据配置对 UAV 排序。"""
        if self.config.uav_sorting == "uav_id_asc":
            return sorted(uavs, key=lambda uav: uav.uav_id)

        if self.config.uav_sorting == "uav_id_desc":
            return sorted(uavs, key=lambda uav: uav.uav_id, reverse=True)

        raise ResourceAllocationError(
            f"Unsupported uav_sorting: {self.config.uav_sorting}"
        )

    @staticmethod
    def _take_uavs(
        available_uavs: list[UAV],
        count: int,
    ) -> list[UAV]:
        """
        从 available_uavs 中取出 count 架 UAV。

        这个函数会原地删除已分配 UAV。
        """
        if count <= 0:
            return []

        taken = available_uavs[:count]
        del available_uavs[:count]

        return taken


def load_resource_allocation_config(
    config: dict[str, Any],
) -> ResourceAllocationConfig:
    """从项目总配置中读取资源分配配置。"""
    prefix = "resource_allocation"

    base_position_raw = get_config_value(
        config,
        f"{prefix}.base_position",
        default=[0.0, -16000.0],
    )
    monte_carlo_config = MonteCarloResourceEvaluatorConfig(
        num_simulations=int(
            get_config_value(
                config,
                "monte_carlo.num_simulations",
                default=get_config_value(
                    config,
                    f"{prefix}.monte_carlo.num_simulations",
                    default=300,
                ),
            )
        ),
        completion_threshold=float(
            get_config_value(
                config,
                "monte_carlo.completion_threshold",
                default=get_config_value(
                    config,
                    f"{prefix}.monte_carlo.completion_threshold",
                    default=0.8,
                ),
            )
        ),
        uav_loss_probability=float(
            get_config_value(
                config,
                "monte_carlo.uav_loss_probability",
                default=get_config_value(
                    config,
                    f"{prefix}.monte_carlo.uav_loss_probability",
                    default=0.05,
                ),
            )
        ),
        target_defense_noise=float(
            get_config_value(
                config,
                "monte_carlo.target_defense_noise",
                default=get_config_value(
                    config,
                    f"{prefix}.monte_carlo.target_defense_noise",
                    default=0.1,
                ),
            )
        ),
        hit_probability_noise=float(
            get_config_value(
                config,
                "monte_carlo.hit_probability_noise",
                default=get_config_value(
                    config,
                    f"{prefix}.monte_carlo.hit_probability_noise",
                    default=0.05,
                ),
            )
        ),
        random_seed=int(
            get_config_value(config, "monte_carlo.random_seed", default=42)
        ),
        output_csv_path=str(
            get_config_value(
                config,
                "monte_carlo.output_csv_path",
                default="outputs/resource_assessment/monte_carlo_results.csv",
            )
        ),
        figure_path=str(
            get_config_value(
                config,
                "monte_carlo.figure_path",
                default=(
                    "outputs/resource_assessment/"
                    "monte_carlo_completion_distribution.png"
                ),
            )
        ),
    )

    allocation_config = ResourceAllocationConfig(
        method=str(
            get_config_value(config, f"{prefix}.method", default="rule_based")
        ),
        attack_payload_per_uav=float(
            get_config_value(
                config,
                f"{prefix}.attack_payload_per_uav",
                default=6.0,
            )
        ),
        guide_uavs_per_cluster=int(
            get_config_value(
                config,
                f"{prefix}.guide_uavs_per_cluster",
                default=1,
            )
        ),
        communication_uavs_per_cluster=int(
            get_config_value(
                config,
                f"{prefix}.communication_uavs_per_cluster",
                default=0,
            )
        ),
        allow_partial_allocation=bool(
            get_config_value(
                config,
                f"{prefix}.allow_partial_allocation",
                default=True,
            )
        ),
        enable_monte_carlo_assessment=bool(
            get_config_value(
                config,
                f"{prefix}.enable_monte_carlo_assessment",
                default=get_config_value(
                    config,
                    "algorithms.resource_assessment.method",
                    default="monte_carlo",
                )
                == "monte_carlo",
            )
        ),
        monte_carlo_config=monte_carlo_config,
        cluster_priority=str(
            get_config_value(
                config,
                f"{prefix}.cluster_priority",
                default="defense_desc",
            )
        ),
        uav_sorting=str(
            get_config_value(
                config,
                f"{prefix}.uav_sorting",
                default="uav_id_asc",
            )
        ),
        base_position=Position(
            x=float(base_position_raw[0]),
            y=float(base_position_raw[1]),
        ),
        debug_csv_path=str(
            get_config_value(
                config,
                f"{prefix}.output.debug_csv_path",
                default="outputs/intermediate/resource_allocation.csv",
            )
        ),
    )

    allocation_config.validate()
    return allocation_config


def build_resource_status_from_state(state) -> ResourceStatus:
    """
    从 BattlefieldState 中构造 ResourceStatus。

    优先使用 state 中已经定义好的 attack_uavs。
    其他 UAV 暂时归入 guide_uavs，通信 UAV 后续可以根据 UAV 类型进一步细分。

    这个函数是当前工程阶段的实用适配层：
    - 如果 entities.py 中已经提供 attack_uavs，就直接使用；
    - 如果后面补充 guide_uavs / communication_uavs，也可以在这里扩展。
    """
    all_uavs = list(state.uavs)

    attack_uavs = list(getattr(state, "attack_uavs", []))

    attack_ids = {uav.uav_id for uav in attack_uavs}

    remaining_uavs = [
        uav for uav in all_uavs
        if uav.uav_id not in attack_ids
    ]

    guide_uavs = list(getattr(state, "guide_uavs", []))
    communication_uavs = list(getattr(state, "communication_uavs", []))

    if not guide_uavs:
        guide_uavs = remaining_uavs

    if not communication_uavs:
        communication_uavs = []

    return ResourceStatus(
        all_uavs=all_uavs,
        available_attack_uavs=attack_uavs,
        available_guide_uavs=guide_uavs,
        available_communication_uavs=communication_uavs,
        damaged_uavs=[],
        metadata={
            "source": "build_resource_status_from_state",
            "num_all_uavs": len(all_uavs),
            "num_attack_uavs": len(attack_uavs),
            "num_guide_uavs": len(guide_uavs),
            "num_communication_uavs": len(communication_uavs),
        },
    )
