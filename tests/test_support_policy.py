"""test支援策略测试模块，用于验证对应业务模块的关键行为。"""
from uav_dynamic_task_allocation.core.contracts import (
    AllocationPlan,
    ClusterAssignment,
    MissionEvent,
    MissionEventType,
    TargetCluster,
)
from uav_dynamic_task_allocation.core.entities import Position, Target, UAV, UAVType
from uav_dynamic_task_allocation.planning.support_policy import (
    SupportPolicy,
    SupportPolicyConfig,
)


def _assignment(cluster_id: int, uav_id: int, completion: float):
    """处理assignment 数据相关业务逻辑。

    参数：
        cluster_id: 目标簇编号，类型为 int。
        uav_id: 无人机编号，类型为 int。
        completion: completion 数据，类型为 float。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    target = Target(cluster_id + 1, Position(float(cluster_id), 0), 1, 1, 1)
    uav = UAV(uav_id, Position(0, 0), UAVType.ATTACK, 100, 5)
    cluster = TargetCluster(cluster_id, [target], target.position, 1, 1)
    return ClusterAssignment(
        cluster_id=cluster_id,
        target_cluster=cluster,
        assigned_attack_uavs=[uav],
        assigned_attack_uav_count=1,
        required_attack_uav_count=1,
        metadata={"resource_assessment": {"completion_probability": completion}},
    )


def test_support_policy_produces_fire_support_decision():
    """处理test支援策略producesfire支援决策相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    receiver = _assignment(0, 1, 0.3)
    donor = _assignment(1, 2, 0.95)
    plan = AllocationPlan(assignments=[receiver, donor])
    event = MissionEvent(
        event_time=1,
        event_type=MissionEventType.ATTACK_UAV_DESTROYED,
        affected_uav_ids=[1],
    )

    decision = SupportPolicy(
        SupportPolicyConfig(donor_min_completion_probability=0.9)
    ).decide_and_apply(event, plan)

    assert decision.support_required
    assert decision.support_type == "fire_support"
    assert decision.receiver_cluster_id == 0
