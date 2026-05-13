from __future__ import annotations

import csv

import numpy as np

from uav_dynamic_task_allocation.allocation.ppo_clusterer import (
    PPOClusterer,
    PPOClustererConfig,
)
from uav_dynamic_task_allocation.allocation.pso_clusterer import PSOClusterer, PSOClustererConfig
from uav_dynamic_task_allocation.envs.target_grouping_env import (
    TargetGroupingEnv,
    TargetGroupingEnvConfig,
)
from uav_dynamic_task_allocation.utils.config import get_project_root, load_and_validate_config


def main() -> None:
    project_root = get_project_root()
    config = load_and_validate_config(project_root / "configs" / "default.yaml")
    ppo_config = config.get("ppo", {})
    checkpoint_dir = project_root / ppo_config.get("checkpoint_dir", "checkpoints/ppo_clusterer")

    features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.9, 1.0],
            [1.0, 0.9],
            [0.45, 0.5],
            [0.52, 0.48],
        ],
        dtype=float,
    )
    values = np.asarray([8, 7, 5, 6, 10, 9], dtype=float)
    defenses = np.asarray([3, 2, 2, 3, 4, 5], dtype=float)
    initial_labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    env = TargetGroupingEnv(
        features=features,
        labels=initial_labels,
        config=TargetGroupingEnvConfig(num_clusters=2, max_targets_per_cluster=8),
        target_values=values,
        target_defenses=defenses,
    )

    ppo_result = PPOClusterer(
        PPOClustererConfig(
            checkpoint_dir=str(checkpoint_dir),
            pso_config=PSOClustererConfig(random_seed=int(config.get("project", {}).get("seed", 42))),
        )
    ).cluster(
        features=features,
        num_clusters=2,
        target_values=values,
        target_defenses=defenses,
    )
    pso_result = PSOClusterer(PSOClustererConfig()).cluster(
        features=features,
        num_clusters=2,
        target_values=values,
        target_defenses=defenses,
    )

    output_path = project_root / "outputs" / "evaluation" / "ppo_clusterer_evaluation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "method": "initial_env",
            "method_used": "initial_labels",
            "score": env.get_metrics()["score"],
            "compactness": env.get_metrics()["compactness"],
            "count_balance": env.get_metrics()["count_balance"],
            "workload_balance": env.get_metrics()["workload_balance"],
            "labels": initial_labels.tolist(),
        },
        {
            "method": "ppo_dynamic",
            "method_used": ppo_result.method_used,
            "score": ppo_result.metrics.get("fitness", ""),
            "compactness": ppo_result.metrics.get("mean_intra_cluster_distance", ""),
            "count_balance": ppo_result.metrics.get("cluster_size_std", ""),
            "workload_balance": ppo_result.metrics.get("task_balance_index", ""),
            "labels": ppo_result.labels.tolist(),
        },
        {
            "method": "pso_static_reference",
            "method_used": "pso",
            "score": pso_result.metrics.get("fitness", ""),
            "compactness": pso_result.metrics.get("mean_intra_cluster_distance", ""),
            "count_balance": pso_result.metrics.get("cluster_size_std", ""),
            "workload_balance": pso_result.metrics.get("task_balance_index", ""),
            "labels": pso_result.labels.tolist(),
        },
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"PPO clusterer evaluation saved to: {output_path}")


if __name__ == "__main__":
    main()
