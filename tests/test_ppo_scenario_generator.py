from uav_dynamic_task_allocation.core.contracts import TargetScore
from uav_dynamic_task_allocation.core.entities import Position, Target
from uav_dynamic_task_allocation.training.ppo_scenario_generator import (
    PPOGroupingScenarioGenerator,
    PPOGroupingScenarioGeneratorConfig,
)


def _targets(count: int, start_id: int = 1) -> list[Target]:
    return [
        Target(
            target_id=start_id + index,
            position=Position(x=float(index), y=float(index % 3)),
            target_type=2,
            defense=1.0 + index % 2,
            significance=2.0 + index % 3,
        )
        for index in range(count)
    ]


def test_ppo_scenario_generator_builds_valid_episode():
    destroy_targets = _targets(6)
    non_destroy_targets = _targets(3, start_id=100)
    target_scores = {
        target.target_id: TargetScore(target_id=target.target_id, score=0.5)
        for target in destroy_targets + non_destroy_targets
    }

    generator = PPOGroupingScenarioGenerator(
        destroy_targets=destroy_targets,
        non_destroy_targets=non_destroy_targets,
        target_scores=target_scores,
        config=PPOGroupingScenarioGeneratorConfig(
            num_clusters=3,
            random_seed=7,
            probabilities={"mixed": 1.0},
        ),
    )

    scenario = generator.generate_episode_scenario(episode_id=1)

    assert scenario.event_type in generator.SUPPORTED_EVENTS
    assert scenario.features.shape[0] == len(scenario.targets)
    assert len(scenario.initial_labels) == len(scenario.targets)
    assert len(scenario.targets) >= 3
    assert scenario.metadata["num_targets"] == len(scenario.targets)
