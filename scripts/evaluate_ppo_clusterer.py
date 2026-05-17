"""Evaluate trained PPO dynamic regrouping against PSO baseline."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from uav_dynamic_task_allocation.algorithms.rl.ppo.visualization import (
    plot_cluster_map,
    plot_ppo_vs_pso_metrics,
)
from uav_dynamic_task_allocation.allocation.ppo_clusterer import (
    PPOClusterer,
    PPOClustererConfig,
)
from uav_dynamic_task_allocation.allocation.pso_clusterer import (
    PSOClusterer,
    PSOClustererConfig,
)
from uav_dynamic_task_allocation.core.entities import Target, build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.envs.target_grouping_env import (
    TargetGroupingEnv,
    TargetGroupingEnvConfig,
)
from uav_dynamic_task_allocation.preprocessing.destroy_target_selection import (
    DestroyTargetSelector,
    load_destroy_target_selection_config,
)
from uav_dynamic_task_allocation.preprocessing.target_screening import (
    TargetScreener,
    load_target_screening_config,
)
from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    get_project_root,
    load_and_validate_config,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO vs PSO target clustering.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_and_validate_config(config_path)

    data = load_all_data(config)
    state = build_battlefield_state(data["uav"], data["target"])
    screened_set = TargetScreener(load_target_screening_config(config)).screen(
        state.targets
    )
    selection_result = DestroyTargetSelector(
        load_destroy_target_selection_config(config)
    ).select(screened_set)
    selected_set = selection_result.to_screened_target_set()
    targets = selection_result.destroy_targets

    feature_names = list(
        get_config_value(
            config,
            "ppo_clusterer_training.feature_names",
            get_config_value(config, "target_clustering.feature_names", ["x", "y"]),
        )
    )
    raw_features = build_feature_matrix(targets, selected_set, feature_names)
    features = normalize_features(raw_features)
    xy_features = np.asarray(
        [[target.position.x, target.position.y] for target in targets],
        dtype=np.float64,
    )
    values = np.asarray([float(target.significance) for target in targets], dtype=np.float64)
    defenses = np.asarray([float(target.defense) for target in targets], dtype=np.float64)
    num_clusters = int(get_config_value(config, "target_clustering.num_clusters", 3))
    pso_config = load_pso_config(config)

    pso_result = PSOClusterer(pso_config).cluster(
        features=features,
        num_clusters=num_clusters,
        target_values=values,
        target_defenses=defenses,
    )
    ppo_result = PPOClusterer(
        PPOClustererConfig(
            checkpoint_dir=str(
                resolve_path(
                    get_config_value(
                        config,
                        "target_clustering.ppo.checkpoint_dir",
                        "checkpoints/ppo_clusterer",
                    ),
                    project_root=project_root,
                )
            ),
            fallback_method=str(
                get_config_value(config, "target_clustering.ppo.fallback_method", "pso")
            ),
            max_inference_steps=int(
                get_config_value(config, "target_clustering.ppo.max_inference_steps", 30)
            ),
            inference_mode=str(
                get_config_value(config, "target_clustering.ppo.inference_mode", "greedy")
            ),
            max_targets_per_cluster=int(
                get_config_value(
                    config,
                    "target_clustering.ppo.max_targets_per_cluster",
                    get_config_value(
                        config,
                        "ppo_clusterer_training.env.max_targets_per_cluster",
                        64,
                    ),
                )
            ),
            random_seed=int(get_config_value(config, "ppo_clusterer_training.random_seed", 42)),
            pso_config=pso_config,
        )
    ).cluster(
        features=features,
        num_clusters=num_clusters,
        target_values=values,
        target_defenses=defenses,
    )

    rows = [
        build_metric_row("pso_baseline", features, pso_result.labels, values, defenses, num_clusters),
        build_metric_row(
            ppo_result.method_used,
            features,
            ppo_result.labels,
            values,
            defenses,
            num_clusters,
        ),
    ]
    rows[1]["fallback_reason"] = ppo_result.fallback_reason or ""
    rows[1]["num_inference_steps"] = ppo_result.num_inference_steps

    comparison_csv = resolve_path(
        "outputs/evaluation/ppo_vs_pso_cluster_metrics.csv",
        project_root=project_root,
    )
    save_csv(rows, comparison_csv)

    figure_dir = resolve_path("outputs/evaluation/figures", project_root=project_root)
    plot_ppo_vs_pso_metrics(
        comparison_csv,
        figure_dir / "ppo_vs_pso_metrics.png",
    )
    plot_cluster_map(
        xy_features,
        ppo_result.labels,
        figure_dir / "ppo_cluster_map.png",
        "PPO target clusters",
    )
    plot_cluster_map(
        xy_features,
        pso_result.labels,
        figure_dir / "pso_cluster_map.png",
        "PSO target clusters",
    )
    print(f"PPO vs PSO metrics saved to: {comparison_csv}")


def build_feature_matrix(
    targets: list[Target],
    screened_set: Any,
    feature_names: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [
                get_feature_value(target, screened_set, feature_name)
                for feature_name in feature_names
            ]
            for target in targets
        ],
        dtype=np.float64,
    )


def get_feature_value(target: Target, screened_set: Any, feature_name: str) -> float:
    if feature_name == "x":
        return float(target.position.x)
    if feature_name == "y":
        return float(target.position.y)
    if feature_name == "defense":
        return float(target.defense)
    if feature_name == "significance":
        return float(target.significance)
    target_score = screened_set.target_scores[int(target.target_id)]
    if feature_name == "score":
        return float(target_score.score)
    if feature_name in target_score.components:
        return float(target_score.components[feature_name])
    if feature_name in target_score.metadata:
        return float(target_score.metadata[feature_name])
    raise ValueError(f"Unsupported feature_name={feature_name}")


def normalize_features(features: np.ndarray) -> np.ndarray:
    min_values = features.min(axis=0)
    max_values = features.max(axis=0)
    ranges = np.where(np.abs(max_values - min_values) < 1e-12, 1.0, max_values - min_values)
    return (features - min_values) / ranges


def load_pso_config(config: dict[str, Any]) -> PSOClustererConfig:
    prefix = "target_clustering.pso"
    return PSOClustererConfig(
        num_particles=int(get_config_value(config, f"{prefix}.num_particles", 30)),
        max_iter=int(get_config_value(config, f"{prefix}.max_iter", 100)),
        inertia_weight=float(get_config_value(config, f"{prefix}.inertia_weight", 0.7)),
        cognitive_weight=float(get_config_value(config, f"{prefix}.cognitive_weight", 1.5)),
        social_weight=float(get_config_value(config, f"{prefix}.social_weight", 1.5)),
        compactness_weight=float(get_config_value(config, f"{prefix}.compactness_weight", 1.0)),
        balance_weight=float(get_config_value(config, f"{prefix}.balance_weight", 0.1)),
        value_weight=float(get_config_value(config, f"{prefix}.value_weight", 0.2)),
        defense_weight=float(get_config_value(config, f"{prefix}.defense_weight", 0.2)),
        convergence_threshold=float(
            get_config_value(config, f"{prefix}.convergence_threshold", 1e-6)
        ),
        random_seed=int(get_config_value(config, f"{prefix}.random_seed", 42)),
    )


def build_metric_row(
    method: str,
    features: np.ndarray,
    labels: np.ndarray,
    values: np.ndarray,
    defenses: np.ndarray,
    num_clusters: int,
) -> dict[str, Any]:
    env = TargetGroupingEnv(
        features=features,
        labels=labels,
        config=TargetGroupingEnvConfig(num_clusters=num_clusters),
        target_values=values,
        target_defenses=defenses,
    )
    env.reset()
    metrics = env.get_metrics()
    return {
        "method": method,
        "final_score": metrics["final_score"],
        "compactness": metrics["compactness"],
        "count_balance": metrics["count_balance"],
        "workload_balance": metrics["workload_balance"],
        "mean_intra_cluster_distance": metrics["mean_intra_cluster_distance"],
        "count_std": metrics["count_std"],
        "workload_std": metrics["workload_std"],
        "empty_cluster_count": metrics["empty_cluster_count"],
        "fallback_reason": "",
        "num_inference_steps": 0,
    }


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
