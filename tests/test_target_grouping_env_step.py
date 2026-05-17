import numpy as np

from uav_dynamic_task_allocation.envs.target_grouping_env import (
    TargetGroupingEnv,
    TargetGroupingEnvConfig,
)


def test_target_grouping_env_step_and_metrics():
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [0.9, 1.0],
            [0.5, 0.5],
            [0.6, 0.5],
        ],
        dtype=float,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    env = TargetGroupingEnv(
        features=features,
        labels=labels,
        config=TargetGroupingEnvConfig(num_clusters=3, max_steps=5),
        target_values=np.asarray([1, 2, 3, 4, 5, 6], dtype=float),
        target_defenses=np.asarray([2, 2, 1, 1, 3, 3], dtype=float),
    )

    observation = env.reset()
    assert observation.shape == (env.observation_dim,)

    mask = env.action_mask()
    assert mask.sum() > 0

    action_id = int(np.where(mask > 0)[0][0])
    _, _, _, info = env.step_index(action_id)

    assert info["legal_action"] is True
    assert len(env.get_current_labels()) == len(features)

    metrics = env.get_metrics()
    for key in ["final_score", "compactness", "count_balance", "workload_balance"]:
        assert key in metrics
