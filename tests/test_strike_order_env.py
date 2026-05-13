from uav_dynamic_task_allocation.core.contracts import TargetCluster
from uav_dynamic_task_allocation.core.entities import Position, Target
from uav_dynamic_task_allocation.envs.strike_order_env import (
    StrikeOrderEnv,
    StrikeOrderEnvConfig,
)


def test_strike_order_env_masks_visited_target():
    targets = [
        Target(1, Position(1, 0), 1, defense=1, significance=2),
        Target(2, Position(2, 0), 1, defense=1, significance=1),
    ]
    cluster = TargetCluster(0, targets, Position(0, 0), 2, 3)
    env = StrikeOrderEnv(
        target_cluster=cluster,
        start_position=cluster.center,
        config=StrikeOrderEnvConfig(max_targets_per_cluster=4),
    )

    observation = env.reset()
    assert observation.action_mask[:2].sum() == 2
    result = env.step(0)
    assert result.observation.action_mask[0] == 0
