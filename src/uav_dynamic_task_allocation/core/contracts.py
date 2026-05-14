"""core 数据模块中的contracts 数据实现。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from uav_dynamic_task_allocation.core.entities import Position, Target, UAV


class ContractError(Exception):
    """全流程数据合同构造、校验和跨模块传递过程中的自定义错误。"""


class PipelineStage(str, Enum):
    """
    任务级仿真流程阶段。

    使用枚举而不是字符串，是为了避免后面模块之间传递阶段名称时出现拼写错误。
    例如 target_screening、target_clustering、resource_allocation 等阶段，
    后面都可以在日志、metrics 和 event 中统一引用。
    """

    # DATA_INITIALIZATION: 数据initialization。
    DATA_INITIALIZATION = "data_initialization"
    # TARGET_SCREENING: 目标screening。
    TARGET_SCREENING = "target_screening"
    # TARGET_CLUSTERING: 目标clustering。
    TARGET_CLUSTERING = "target_clustering"
    # RESOURCE_ALLOCATION: 资源资源分配。
    RESOURCE_ALLOCATION = "resource_allocation"
    # STRIKE_ORDER_PLANNING: 打击顺序planning。
    STRIKE_ORDER_PLANNING = "strike_order_planning"
    # MISSION_SIMULATION: 任务仿真。
    MISSION_SIMULATION = "mission_simulation"
    # DYNAMIC_REPLANNING: 动态重规划。
    DYNAMIC_REPLANNING = "dynamic_replanning"
    # EVALUATION: 评估。
    EVALUATION = "evaluation"


class MissionEventType(str, Enum):
    """
    任务仿真中的动态事件类型。

    这些事件对应源代码和论文中提到的动态战场变化：
    目标消失、新目标出现、攻击无人机损毁、导引/侦察无人机损毁等。
    后续 simulation/dynamic_events.py 会使用这些事件类型来触发重规划。
    """

    # TARGET_APPEARED: 目标appeared。
    TARGET_APPEARED = "target_appeared"
    # TARGET_DISAPPEARED: 目标disappeared。
    TARGET_DISAPPEARED = "target_disappeared"
    # ATTACK_UAV_DESTROYED: 攻击无人机毁伤状态。
    ATTACK_UAV_DESTROYED = "attack_uav_destroyed"
    # GUIDE_UAV_DESTROYED: 导引无人机毁伤状态。
    GUIDE_UAV_DESTROYED = "guide_uav_destroyed"
    # COMMUNICATION_UAV_DESTROYED: 通信无人机毁伤状态。
    COMMUNICATION_UAV_DESTROYED = "communication_uav_destroyed"
    # RESOURCE_SHORTAGE: 资源shortage。
    RESOURCE_SHORTAGE = "resource_shortage"
    # REPLANNING_TRIGGERED: 重规划triggered。
    REPLANNING_TRIGGERED = "replanning_triggered"
    # CUSTOM: CUSTOM 数据。
    CUSTOM = "custom"


@dataclass(frozen=True)
class AlgorithmMetadata:
    """
    算法执行元信息。

    这个结构用于增强算法兼容性。
    不管后面使用 PSO、PPO、KMeans、DQN、Greedy 还是其他算法，
    都可以把算法名称、版本、配置和运行信息放在 metadata 中。

    这样后续做实验对比时，不需要猜某个结果来自哪个算法。
    """

    # algorithm_name: algorithm名称。
    algorithm_name: str
    # algorithm_type: algorithm类型。
    algorithm_type: str
    # version: version 数据。
    version: str = "v1"
    # stage: 阶段。
    stage: PipelineStage | None = None
    # config: 配置。
    config: dict[str, Any] = field(default_factory=dict)
    # runtime_seconds: 运行时秒数。
    runtime_seconds: float | None = None
    # notes: notes 数据。
    notes: str = ""


@dataclass
class TargetScore:
    """
    单个目标的筛选评分结果。

    target_id:
        目标编号。

    score:
        综合评分，可以由重要性、防御值、类型权重、威胁度等共同计算。

    components:
        评分分项。例如：
        {
            "significance": 0.6,
            "defense": 0.2,
            "type_weight": 0.2
        }

    这样设计的好处是：以后你换目标筛选算法时，只要仍然输出 TargetScore，
    后面的目标分群和资源分配模块就不用改。
    """

    # target_id: 目标编号。
    target_id: int
    # score: 评分。
    score: float
    # components: components 数据。
    components: dict[str, float] = field(default_factory=dict)
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreenedTargetSet:
    """
    目标筛选阶段输出。

    该对象对应源代码中 Target_Screen / Kmeans_step1 一类功能的统一输出。

    all_targets:
        原始全部目标。

    candidate_targets:
        经过初步筛选后认为值得考虑的目标。

    selected_targets:
        最终进入摧毁目标集或任务目标集的目标。

    target_scores:
        每个目标的评分结果。

    metadata:
        保存筛选算法、阈值、筛选原因等扩展信息。
    """

    # all_targets: all目标集合。
    all_targets: list[Target]
    # candidate_targets: candidate目标集合。
    candidate_targets: list[Target]
    # selected_targets: 已选择目标集合。
    selected_targets: list[Target]
    # target_scores: 目标评分集合。
    target_scores: dict[int, TargetScore] = field(default_factory=dict)
    # algorithm_metadata: algorithm扩展元数据。
    algorithm_metadata: AlgorithmMetadata | None = None
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """检查筛选结果是否合理。"""
        all_ids = {target.target_id for target in self.all_targets}
        candidate_ids = {target.target_id for target in self.candidate_targets}
        selected_ids = {target.target_id for target in self.selected_targets}

        if not candidate_ids.issubset(all_ids):
            raise ContractError("candidate_targets must be a subset of all_targets.")

        if not selected_ids.issubset(all_ids):
            raise ContractError("selected_targets must be a subset of all_targets.")


@dataclass
class TargetCluster:
    """
    单个目标群。

    这个对象用于承接 PSO / PPO / KMeans / 手工规则等不同目标分群算法的输出。

    cluster_id:
        目标群编号。

    targets:
        属于该群的目标列表。

    center:
        目标群中心。可以由目标坐标均值、加权中心或算法输出中心确定。

    defense_sum:
        该群目标总防御值，资源分配时会用到。

    significance_sum:
        该群目标总重要性，用于任务优先级评估。

    compactness:
        群内紧凑度，可用于评价分群质量。没有计算时可以为 None。
    """

    # cluster_id: 目标簇编号。
    cluster_id: int
    # targets: 目标集合。
    targets: list[Target]
    # center: center 数据。
    center: Position
    # defense_sum: 防御能力sum。
    defense_sum: float
    # significance_sum: 重要程度sum。
    significance_sum: float
    # compactness: compactness 数据。
    compactness: float | None = None
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_ids(self) -> list[int]:
        """返回该目标群内所有目标编号。"""
        return [target.target_id for target in self.targets]

    @property
    def num_targets(self) -> int:
        """返回该目标群内目标数量。"""
        return len(self.targets)

    def validate(self) -> None:
        """检查目标群是否有效。"""
        if self.cluster_id < 0:
            raise ContractError("cluster_id must be non-negative.")

        if not self.targets:
            raise ContractError(f"TargetCluster {self.cluster_id} has no targets.")


@dataclass
class TargetClusterSet:
    """
    目标分群阶段输出。

    这个对象对应源代码中的 clustered_data、cluster_centers 等分散变量，
    但这里将其包装成统一结构，便于后续资源分配、DQN 打击排序和任务仿真调用。
    """

    # clusters: 目标簇集合。
    clusters: list[TargetCluster]
    # algorithm_metadata: algorithm扩展元数据。
    algorithm_metadata: AlgorithmMetadata | None = None
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_clusters(self) -> int:
        """返回目标群数量。"""
        return len(self.clusters)

    @property
    def all_target_ids(self) -> list[int]:
        """返回所有目标群中的目标编号。"""
        ids: list[int] = []
        for cluster in self.clusters:
            ids.extend(cluster.target_ids)
        return ids

    def get_cluster_by_id(self, cluster_id: int) -> TargetCluster:
        """根据 cluster_id 获取目标群。"""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster

        raise ContractError(f"Cluster not found: cluster_id={cluster_id}")

    def validate(self) -> None:
        """
        检查目标群集合是否合法。

        当前主要检查：
        1. 至少有一个 cluster；
        2. cluster_id 不重复；
        3. 同一个 target 不应同时出现在多个 cluster 中。
        """
        if not self.clusters:
            raise ContractError("TargetClusterSet must contain at least one cluster.")

        cluster_ids = [cluster.cluster_id for cluster in self.clusters]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ContractError("Duplicate cluster_id found in TargetClusterSet.")

        all_target_ids = self.all_target_ids
        if len(all_target_ids) != len(set(all_target_ids)):
            raise ContractError(
                "Duplicate target_id found across different target clusters."
            )

        for cluster in self.clusters:
            cluster.validate()


@dataclass
class ResourceStatus:
    """
    UAV 资源状态。

    这个对象用于描述当前可用 UAV 资源池。
    后续动态事件发生后，例如 UAV 损毁、资源不足、支援补充，
    都可以更新这个对象。
    """

    # all_uavs: alluavs。
    all_uavs: list[UAV]
    # available_guide_uavs: available导引uavs。
    available_guide_uavs: list[UAV] = field(default_factory=list)
    # available_attack_uavs: available攻击uavs。
    available_attack_uavs: list[UAV] = field(default_factory=list)
    # available_communication_uavs: available通信uavs。
    available_communication_uavs: list[UAV] = field(default_factory=list)
    # damaged_uavs: damageduavs。
    damaged_uavs: list[UAV] = field(default_factory=list)
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available_attack_uav_ids(self) -> list[int]:
        """返回当前可用攻击 UAV 编号。"""
        return [uav.uav_id for uav in self.available_attack_uavs]

    @property
    def available_guide_uav_ids(self) -> list[int]:
        """返回当前可用导引 UAV 编号。"""
        return [uav.uav_id for uav in self.available_guide_uavs]


@dataclass
class ClusterAssignment:
    """
    单个目标群的 UAV 资源分配结果。

    该对象对应原 main.py 中 allocation_result 的结构化版本。

    assigned_attack_uavs:
        分配给该目标群的攻击 UAV。

    assigned_guide_uavs:
        分配给该目标群的导引/侦察 UAV。

    start_position:
        该 UAV 小组出发位置。可以是基地，也可以是动态事件后的当前位置。

    real_time_position:
        仿真时间推进过程中的实时位置。MissionSimulator 会更新它。
    """

    # cluster_id: 目标簇编号。
    cluster_id: int
    # target_cluster: 目标目标簇。
    target_cluster: TargetCluster
    # assigned_attack_uavs: assigned攻击uavs。
    assigned_attack_uavs: list[UAV]
    # assigned_guide_uavs: assigned导引uavs。
    assigned_guide_uavs: list[UAV] = field(default_factory=list)
    # assigned_communication_uavs: assigned通信uavs。
    assigned_communication_uavs: list[UAV] = field(default_factory=list)

    # required_attack_uav_count: required攻击无人机count。
    required_attack_uav_count: int = 0
    # assigned_attack_uav_count: assigned攻击无人机count。
    assigned_attack_uav_count: int = 0

    # start_position: start位置坐标。
    start_position: Position | None = None
    # real_time_position: real时间位置坐标。
    real_time_position: Position | None = None
    # target_center: 目标center。
    target_center: Position | None = None

    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def assigned_attack_uav_ids(self) -> list[int]:
        """返回分配的攻击 UAV 编号。"""
        return [uav.uav_id for uav in self.assigned_attack_uavs]

    @property
    def assigned_guide_uav_ids(self) -> list[int]:
        """返回分配的导引 UAV 编号。"""
        return [uav.uav_id for uav in self.assigned_guide_uavs]

    def validate(self) -> None:
        """检查资源分配结果是否合理。"""
        if self.cluster_id != self.target_cluster.cluster_id:
            raise ContractError(
                "ClusterAssignment.cluster_id must match target_cluster.cluster_id."
            )

        if self.assigned_attack_uav_count != len(self.assigned_attack_uavs):
            raise ContractError(
                "assigned_attack_uav_count must equal len(assigned_attack_uavs)."
            )


@dataclass
class AllocationPlan:
    """
    资源分配阶段输出。

    一个 AllocationPlan 包含多个 ClusterAssignment。
    后续 MissionSimulator 会根据它知道每个 UAV 小组要飞向哪个目标群。
    """

    # assignments: assignments 数据。
    assignments: list[ClusterAssignment]
    # resource_status: 资源状态。
    resource_status: ResourceStatus | None = None
    # algorithm_metadata: algorithm扩展元数据。
    algorithm_metadata: AlgorithmMetadata | None = None
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_assignments(self) -> int:
        """返回分配任务数量。"""
        return len(self.assignments)

    def get_assignment_by_cluster_id(self, cluster_id: int) -> ClusterAssignment:
        """根据 cluster_id 获取分配结果。"""
        for assignment in self.assignments:
            if assignment.cluster_id == cluster_id:
                return assignment

        raise ContractError(f"Assignment not found: cluster_id={cluster_id}")

    def validate(self) -> None:
        """检查分配计划是否合法。"""
        cluster_ids = [assignment.cluster_id for assignment in self.assignments]

        if len(cluster_ids) != len(set(cluster_ids)):
            raise ContractError("Duplicate cluster assignment found.")

        for assignment in self.assignments:
            assignment.validate()


@dataclass
class StrikeOrderPlan:
    """
    目标群内部打击次序规划结果。

    这个对象对应原代码 attack_order_dqn.py 的输出。
    后续可以由 DQN、Greedy、ACO、TSP、Rule-based 等不同算法生成。
    """

    # cluster_id: 目标簇编号。
    cluster_id: int
    # ordered_target_ids: ordered目标编号集合。
    ordered_target_ids: list[int]
    # path_positions: 路径positions。
    path_positions: list[Position] = field(default_factory=list)
    # total_path_distance: total路径distance。
    total_path_distance: float | None = None
    # expected_reward: expected奖励。
    expected_reward: float | None = None
    # repeated_target_count: repeated目标count。
    repeated_target_count: int = 0
    # algorithm_metadata: algorithm扩展元数据。
    algorithm_metadata: AlgorithmMetadata | None = None
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """检查打击次序是否合法。"""
        if len(self.ordered_target_ids) != len(set(self.ordered_target_ids)):
            raise ContractError(
                f"StrikeOrderPlan for cluster {self.cluster_id} "
                "contains duplicated target ids."
            )


@dataclass
class MissionPlan:
    """
    完整任务规划结果。

    MissionPlan 是任务仿真前的核心输入。
    它组合了目标筛选、目标分群、资源分配和打击排序结果。

    后续 MissionSimulator 不应该重新猜算法输出，
    而是直接读取 MissionPlan 推进时间线。
    """

    # screened_targets: 筛选结果目标集合。
    screened_targets: ScreenedTargetSet
    # target_clusters: 目标目标簇集合。
    target_clusters: TargetClusterSet
    # allocation_plan: 资源分配规划方案。
    allocation_plan: AllocationPlan
    # strike_order_plans: 打击顺序plans。
    strike_order_plans: dict[int, StrikeOrderPlan] = field(default_factory=dict)
    # algorithm_metadata: algorithm扩展元数据。
    algorithm_metadata: list[AlgorithmMetadata] = field(default_factory=list)
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        检查 MissionPlan 内部引用是否一致。

        主要检查：
        1. cluster set 合法；
        2. allocation plan 合法；
        3. strike_order_plans 中的 cluster_id 必须存在于 target_clusters。
        """
        self.screened_targets.validate()
        self.target_clusters.validate()
        self.allocation_plan.validate()

        valid_cluster_ids = {cluster.cluster_id for cluster in self.target_clusters.clusters}
        strike_cluster_ids = set(self.strike_order_plans.keys())

        if not strike_cluster_ids.issubset(valid_cluster_ids):
            raise ContractError(
                "strike_order_plans contains cluster ids not found in target_clusters."
            )


@dataclass
class MissionEvent:
    """
    任务仿真中的动态事件记录。

    event_time:
        事件发生时间。

    event_type:
        事件类型。

    affected_target_ids / affected_uav_ids:
        事件影响的目标或 UAV。

    payload:
        扩展信息。例如新目标信息、损毁原因、重规划原因等。
    """

    # event_time: 事件时间。
    event_time: float
    # event_type: 事件类型。
    event_type: MissionEventType
    # affected_target_ids: affected目标编号集合。
    affected_target_ids: list[int] = field(default_factory=list)
    # affected_uav_ids: affected无人机编号集合。
    affected_uav_ids: list[int] = field(default_factory=list)
    # payload: payload 数据。
    payload: dict[str, Any] = field(default_factory=dict)
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionTimeSnapshot:
    """
    任务仿真某个时间点的快照。

    MissionSimulator 每推进一个时间步，可以选择保存一个 snapshot。
    后续用于绘制 UAV 轨迹、目标状态变化、事件响应过程等。
    """

    # time: 时间。
    time: float
    # cluster_positions: 目标簇positions。
    cluster_positions: dict[int, Position] = field(default_factory=dict)
    # active_target_ids: 可用状态目标编号集合。
    active_target_ids: list[int] = field(default_factory=list)
    # destroyed_target_ids: 毁伤状态目标编号集合。
    destroyed_target_ids: list[int] = field(default_factory=list)
    # active_uav_ids: 可用状态无人机编号集合。
    active_uav_ids: list[int] = field(default_factory=list)
    # damaged_uav_ids: damaged无人机编号集合。
    damaged_uav_ids: list[int] = field(default_factory=list)
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionMetrics:
    """
    任务级仿真评估指标。

    这些指标用于最终实验对比，而不是 DQN 训练过程中的 loss。
    """

    # total_targets: total目标集合。
    total_targets: int = 0
    # selected_targets: 已选择目标集合。
    selected_targets: int = 0
    # destroyed_targets: 毁伤状态目标集合。
    destroyed_targets: int = 0
    # high_value_destroyed_targets: high数值毁伤状态目标集合。
    high_value_destroyed_targets: int = 0

    # total_uavs: totaluavs。
    total_uavs: int = 0
    # lost_uavs: lostuavs。
    lost_uavs: int = 0

    # total_distance: totaldistance。
    total_distance: float = 0.0
    # total_reward: total奖励。
    total_reward: float = 0.0
    # mission_completion_rate: 任务completion率。
    mission_completion_rate: float = 0.0

    # num_dynamic_events: num动态事件集合。
    num_dynamic_events: int = 0
    # num_replanning: num重规划。
    num_replanning: int = 0

    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionResult:
    """
    完整任务仿真输出。

    该对象是 run_mission_simulation.py 的最终结果。
    后续可以保存为 JSON、CSV、图表，或者用于多算法对比实验。
    """

    # mission_plan: 标准任务规划方案。
    mission_plan: MissionPlan
    # time_series: 时间series。
    time_series: list[MissionTimeSnapshot]
    # event_log: 事件日志。
    event_log: list[MissionEvent]
    # metrics: 指标集合。
    metrics: MissionMetrics
    # final_target_status: final目标状态。
    final_target_status: dict[int, str] = field(default_factory=dict)
    # final_uav_status: final无人机状态。
    final_uav_status: dict[int, str] = field(default_factory=dict)
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)