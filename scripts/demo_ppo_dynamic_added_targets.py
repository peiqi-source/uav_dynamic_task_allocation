"""Demo PPO dynamic regrouping when new targets are added."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from uav_dynamic_task_allocation.algorithms.rl.ppo.visualization import (
    plot_dynamic_added_targets_before_after,
)
from uav_dynamic_task_allocation.allocation.ppo_clusterer import (
    PPOClusterer,
    PPOClustererConfig,
)
from uav_dynamic_task_allocation.allocation.pso_clusterer import (
    PSOClusterer,
    PSOClustererConfig,
)
from uav_dynamic_task_allocation.core.entities import (
    Position,
    Target,
    build_battlefield_state,
)
from uav_dynamic_task_allocation.data.loaders import load_all_data
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
    parser = argparse.ArgumentParser(
        description="Visual demo for PPO regrouping after adding new targets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--num-new-targets",
        type=int,
        default=2,
        help="Number of newly added targets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for new-target sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_new_targets <= 0:
        raise ValueError("--num-new-targets must be positive.")

    project_root = get_project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_and_validate_config(config_path)
    rng = np.random.default_rng(args.seed)

    data = load_all_data(config)
    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )
    screened_set = TargetScreener(load_target_screening_config(config)).screen(
        state.targets
    )
    selection_result = DestroyTargetSelector(
        load_destroy_target_selection_config(config)
    ).select(screened_set)
    selected_set = selection_result.to_screened_target_set()
    destroy_targets = list(selection_result.destroy_targets)
    non_destroy_targets = list(selection_result.non_destroy_targets)

    feature_names = list(
        get_config_value(
            config,
            "ppo_clusterer_training.feature_names",
            get_config_value(config, "target_clustering.feature_names", ["x", "y"]),
        )
    )
    num_clusters = int(get_config_value(config, "target_clustering.num_clusters", 3))
    pso_config = load_pso_config(config)

    original_raw_features = build_feature_matrix(
        destroy_targets,
        selected_set,
        feature_names,
    )
    original_features = normalize_features(original_raw_features)
    values = np.asarray(
        [float(target.significance) for target in destroy_targets],
        dtype=np.float64,
    )
    defenses = np.asarray(
        [float(target.defense) for target in destroy_targets],
        dtype=np.float64,
    )
    original_result = PSOClusterer(pso_config).cluster(
        features=original_features,
        num_clusters=num_clusters,
        target_values=values,
        target_defenses=defenses,
    )
    original_labels = original_result.labels

    new_targets = sample_or_create_new_targets(
        destroy_targets=destroy_targets,
        non_destroy_targets=non_destroy_targets,
        num_new_targets=args.num_new_targets,
        rng=rng,
    )
    dynamic_targets = destroy_targets + new_targets
    original_count = len(destroy_targets)

    dynamic_raw_features = build_feature_matrix(
        dynamic_targets,
        selected_set,
        feature_names,
    )
    dynamic_features = normalize_features(dynamic_raw_features)
    dynamic_xy_features = np.asarray(
        [[target.position.x, target.position.y] for target in dynamic_targets],
        dtype=np.float64,
    )
    dynamic_values = np.asarray(
        [float(target.significance) for target in dynamic_targets],
        dtype=np.float64,
    )
    dynamic_defenses = np.asarray(
        [float(target.defense) for target in dynamic_targets],
        dtype=np.float64,
    )

    before_labels = assign_new_targets_to_nearest_centers(
        xy_features=dynamic_xy_features,
        original_count=original_count,
        original_labels=original_labels,
        num_clusters=num_clusters,
    )

    ppo_result = PPOClusterer(
        build_ppo_config(config, project_root, pso_config=pso_config)
    ).cluster(
        features=dynamic_features,
        num_clusters=num_clusters,
        target_values=dynamic_values,
        target_defenses=dynamic_defenses,
    )
    if ppo_result.method_used == "ppo_fallback_pso":
        print("Warning: PPO checkpoint unavailable, fallback to PSO.")

    after_labels = ppo_result.labels
    is_new_target = np.asarray(
        [index >= original_count for index in range(len(dynamic_targets))],
        dtype=bool,
    )

    figure_path = resolve_path(
        "outputs/evaluation/figures/ppo_dynamic_added_targets_before_after.png",
        project_root=project_root,
    )
    plot_dynamic_added_targets_before_after(
        xy_features=dynamic_xy_features,
        before_labels=before_labels,
        after_labels=after_labels,
        is_new_target=is_new_target,
        output_path=figure_path,
    )

    csv_path = resolve_path(
        "outputs/evaluation/ppo_dynamic_added_targets_demo.csv",
        project_root=project_root,
    )
    save_demo_csv(
        output_path=csv_path,
        targets=dynamic_targets,
        is_new_target=is_new_target,
        before_labels=before_labels,
        after_labels=after_labels,
        method_used=ppo_result.method_used,
        fallback_reason=ppo_result.fallback_reason or "",
    )
    print(f"Dynamic added-target demo figure saved to: {figure_path}")
    print(f"Dynamic added-target demo CSV saved to: {csv_path}")


def build_feature_matrix(
    targets: list[Target],
    selected_set: Any,
    feature_names: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [
                get_feature_value(target, selected_set, feature_name)
                for feature_name in feature_names
            ]
            for target in targets
        ],
        dtype=np.float64,
    )


def get_feature_value(target: Target, selected_set: Any, feature_name: str) -> float:
    if feature_name == "x":
        return float(target.position.x)
    if feature_name == "y":
        return float(target.position.y)
    if feature_name == "defense":
        return float(target.defense)
    if feature_name == "significance":
        return float(target.significance)
    if feature_name == "target_type":
        return float(
            target.target_type.value
            if hasattr(target.target_type, "value")
            else target.target_type
        )

    target_score = selected_set.target_scores.get(int(target.target_id))
    if target_score is None:
        if feature_name == "score":
            return float(target.significance)
        raise ValueError(f"TargetScore not found for target_id={target.target_id}")
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
    ranges = np.where(
        np.abs(max_values - min_values) < 1e-12,
        1.0,
        max_values - min_values,
    )
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
        initialize_with_high_value_targets=bool(
            get_config_value(
                config,
                f"{prefix}.initialize_with_high_value_targets",
                True,
            )
        ),
    )


def build_ppo_config(
    config: dict[str, Any],
    project_root: Path,
    pso_config: PSOClustererConfig,
) -> PPOClustererConfig:
    checkpoint_dir = resolve_path(
        get_config_value(
            config,
            "target_clustering.ppo.checkpoint_dir",
            "checkpoints/ppo_clusterer",
        ),
        project_root=project_root,
    )
    return PPOClustererConfig(
        checkpoint_dir=str(checkpoint_dir),
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


def sample_or_create_new_targets(
    destroy_targets: list[Target],
    non_destroy_targets: list[Target],
    num_new_targets: int,
    rng: np.random.Generator,
) -> list[Target]:
    sampled_targets: list[Target] = []
    if non_destroy_targets:
        sample_count = min(num_new_targets, len(non_destroy_targets))
        indices = rng.choice(len(non_destroy_targets), size=sample_count, replace=False)
        sampled_targets.extend(non_destroy_targets[int(index)] for index in indices)

    missing_count = num_new_targets - len(sampled_targets)
    if missing_count <= 0:
        return sampled_targets

    all_targets = destroy_targets + sampled_targets
    max_target_id = max(int(target.target_id) for target in all_targets)
    x_values = [float(target.position.x) for target in destroy_targets]
    y_values = [float(target.position.y) for target in destroy_targets]

    for offset in range(missing_count):
        sampled_targets.append(
            Target(
                target_id=max_target_id + offset + 1,
                position=Position(
                    x=float(rng.uniform(min(x_values), max(x_values))),
                    y=float(rng.uniform(min(y_values), max(y_values))),
                ),
                target_type=2,
                defense=float(rng.uniform(1.0, 5.0)),
                significance=float(rng.uniform(1.0, 5.0)),
                metadata={"is_new_target": True, "synthetic": True},
            )
        )
    return sampled_targets


def assign_new_targets_to_nearest_centers(
    xy_features: np.ndarray,
    original_count: int,
    original_labels: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
    before_labels = np.full(len(xy_features), -1, dtype=np.int64)
    before_labels[:original_count] = np.asarray(original_labels, dtype=np.int64)

    centers = np.zeros((num_clusters, 2), dtype=np.float64)
    for cluster_id in range(num_clusters):
        members = xy_features[:original_count][original_labels == cluster_id]
        if len(members) > 0:
            centers[cluster_id] = members.mean(axis=0)

    for index in range(original_count, len(xy_features)):
        distances = np.linalg.norm(centers - xy_features[index], axis=1)
        before_labels[index] = int(np.argmin(distances))
    return before_labels


def save_demo_csv(
    output_path: Path,
    targets: list[Target],
    is_new_target: np.ndarray,
    before_labels: np.ndarray,
    after_labels: np.ndarray,
    method_used: str,
    fallback_reason: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_id",
        "x",
        "y",
        "is_new_target",
        "before_cluster",
        "after_cluster",
        "significance",
        "defense",
        "method_used",
        "fallback_reason",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, target in enumerate(targets):
            writer.writerow(
                {
                    "target_id": int(target.target_id),
                    "x": float(target.position.x),
                    "y": float(target.position.y),
                    "is_new_target": bool(is_new_target[index]),
                    "before_cluster": int(before_labels[index]),
                    "after_cluster": int(after_labels[index]),
                    "significance": float(target.significance),
                    "defense": float(target.defense),
                    "method_used": method_used,
                    "fallback_reason": fallback_reason,
                }
            )


if __name__ == "__main__":
    main()
