def test_ppo_clusterer_imports():
    from uav_dynamic_task_allocation.algorithms.rl.ppo.agent import PPOAgent
    from uav_dynamic_task_allocation.algorithms.rl.ppo.network import PPOActorCritic
    from uav_dynamic_task_allocation.allocation.ppo_clusterer import PPOClusterer
    from uav_dynamic_task_allocation.envs.target_grouping_env import TargetGroupingEnv
    from uav_dynamic_task_allocation.training.ppo_clusterer_training import (
        run_ppo_clusterer_training,
    )
    from uav_dynamic_task_allocation.training.ppo_scenario_generator import (
        PPOGroupingScenarioGenerator,
    )

    assert PPOAgent is not None
    assert PPOActorCritic is not None
    assert PPOClusterer is not None
    assert TargetGroupingEnv is not None
    assert run_ppo_clusterer_training is not None
    assert PPOGroupingScenarioGenerator is not None
