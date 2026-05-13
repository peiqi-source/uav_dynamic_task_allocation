from uav_dynamic_task_allocation.allocation.monte_carlo_resource_evaluator import (
    MonteCarloResourceEvaluator,
    MonteCarloResourceEvaluatorConfig,
)
from uav_dynamic_task_allocation.core.contracts import ClusterAssignment, TargetCluster
from uav_dynamic_task_allocation.core.entities import Position, Target, UAV, UAVType


def test_monte_carlo_resource_evaluator_assesses_assignment():
    target = Target(1, Position(10, 0), 1, defense=1.0, significance=2.0)
    uav = UAV(1, Position(0, 0), UAVType.ATTACK, work_range=100, attack_power=10)
    cluster = TargetCluster(0, [target], Position(10, 0), 1.0, 2.0)
    assignment = ClusterAssignment(
        cluster_id=0,
        target_cluster=cluster,
        assigned_attack_uavs=[uav],
        assigned_attack_uav_count=1,
        required_attack_uav_count=1,
    )

    result = MonteCarloResourceEvaluator(
        MonteCarloResourceEvaluatorConfig(num_simulations=20, random_seed=1)
    ).assess_assignment(assignment)

    assert result.cluster_id == 0
    assert 0.0 <= result.completion_probability <= 1.0
