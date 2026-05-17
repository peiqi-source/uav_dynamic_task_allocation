"""Compare PSO target clustering with different feature-name combinations."""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_dynamic_task_allocation.allocation.pso_clusterer import (
    PSOClusterer,
    PSOClustererConfig,
)
from uav_dynamic_task_allocation.core.entities import Target, build_battlefield_state
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
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


DEFAULT_FEATURE_SETS: dict[str, list[str]] = {
    "xy": ["x", "y"],
    "xy_score": ["x", "y", "score"],
    "xy_score_defense": ["x", "y", "score", "defense"],
    "xy_score_defense_significance": [
        "x",
        "y",
        "score",
        "defense",
        "significance",
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare PSO target clustering across feature sets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file.",
    )
    return parser.parse_args()


def resolve_config_path(config_path: str) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(config_path)
    if path.is_absolute():
        return path
    return get_project_root() / path


def normalize_features(features: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Min-max normalize feature columns for PSO."""
    min_values = features.min(axis=0)
    max_values = features.max(axis=0)
    ranges = max_values - min_values
    safe_ranges = np.where(np.abs(ranges) < 1e-12, 1.0, ranges)
    normalized = (features - min_values) / safe_ranges
    return normalized, {
        "min_values": min_values.tolist(),
        "max_values": max_values.tolist(),
        "ranges": safe_ranges.tolist(),
    }


def get_target_score(screened_set: Any, target: Target) -> float:
    """Return the target screening score used as a clustering feature."""
    target_score = screened_set.target_scores.get(int(target.target_id))
    if target_score is None:
        raise RuntimeError(f"TargetScore not found for target_id={target.target_id}")
    return float(target_score.score)


def get_feature_value(screened_set: Any, target: Target, feature_name: str) -> float:
    """Read one target feature by name."""
    if feature_name == "x":
        return float(target.position.x)
    if feature_name == "y":
        return float(target.position.y)
    if feature_name == "score":
        return get_target_score(screened_set, target)
    if feature_name == "defense":
        return float(target.defense)
    if feature_name == "significance":
        return float(target.significance)
    if feature_name == "target_type":
        target_type = target.target_type
        if hasattr(target_type, "value"):
            return float(target_type.value)
        return float(target_type)

    target_score = screened_set.target_scores.get(int(target.target_id))
    if target_score is not None:
        if feature_name in target_score.components:
            return float(target_score.components[feature_name])
        if feature_name in target_score.metadata:
            return float(target_score.metadata[feature_name])

    raise RuntimeError(
        f"Unsupported feature_name={feature_name}. "
        "Available basic features: x, y, score, defense, significance, target_type."
    )


def build_feature_matrix(
    targets: list[Target],
    screened_set: Any,
    feature_names: list[str],
) -> np.ndarray:
    """Build the PSO input feature matrix."""
    return np.asarray(
        [
            [
                get_feature_value(
                    screened_set=screened_set,
                    target=target,
                    feature_name=feature_name,
                )
                for feature_name in feature_names
            ]
            for target in targets
        ],
        dtype=np.float64,
    )


def calculate_pso_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    num_clusters: int,
    target_values: np.ndarray,
    target_defenses: np.ndarray,
    pso_config: PSOClustererConfig,
) -> dict[str, float]:
    """Calculate comparison metrics from labels and normalized features."""
    distances = np.linalg.norm(features - centers[labels], axis=1)

    sample_weights = np.ones(len(features), dtype=np.float64)
    if pso_config.value_weight > 0:
        sample_weights += pso_config.value_weight * normalize_vector(target_values)
    if pso_config.defense_weight > 0:
        sample_weights += pso_config.defense_weight * normalize_vector(target_defenses)

    compactness = (
        float(np.average(distances, weights=sample_weights))
        if len(distances)
        else 0.0
    )

    counts = np.asarray(
        [np.sum(labels == cluster_id) for cluster_id in range(num_clusters)],
        dtype=np.float64,
    )
    ideal = len(features) / max(num_clusters, 1)
    balance_penalty = float(np.mean((counts - ideal) ** 2) / max(ideal, 1.0))

    return {
        "compactness": compactness,
        "balance_penalty": balance_penalty,
        "empty_cluster_count": float(np.sum(counts == 0)),
        "cluster_size_min": float(np.min(counts)) if len(counts) else 0.0,
        "cluster_size_max": float(np.max(counts)) if len(counts) else 0.0,
        "cluster_size_std": float(np.std(counts)) if len(counts) else 0.0,
    }


def normalize_vector(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D vector."""
    values = np.asarray(values, dtype=np.float64)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if abs(max_value - min_value) < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - min_value) / (max_value - min_value)


def build_detail_rows(
    targets: list[Target],
    screened_set: Any,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Build target-level cluster assignment rows."""
    rows: list[dict[str, Any]] = []
    for target, label in zip(targets, labels):
        rows.append(
            {
                "target_id": int(target.target_id),
                "x": float(target.position.x),
                "y": float(target.position.y),
                "score": get_target_score(screened_set, target),
                "defense": float(target.defense),
                "significance": float(target.significance),
                "cluster_id": int(label),
            }
        )
    return rows


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save rows to CSV, creating parent directories automatically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to save: {output_path}")

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_feature_sets(config: dict[str, Any]) -> dict[str, list[str]]:
    """Read feature-set definitions from config, falling back to defaults."""
    raw_feature_sets = get_config_value(
        config,
        "pso_feature_set_comparison.feature_sets",
        default=DEFAULT_FEATURE_SETS,
    )
    return {
        str(name): [str(feature_name) for feature_name in feature_names]
        for name, feature_names in raw_feature_sets.items()
    }


def get_output_paths(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    """Read comparison output paths from config."""
    return {
        "comparison_csv": resolve_path(
            get_config_value(
                config,
                "pso_feature_set_comparison.output.comparison_csv",
                default="outputs/evaluation/pso_feature_set_comparison.csv",
            ),
            project_root=project_root,
        ),
        "detail_dir": resolve_path(
            get_config_value(
                config,
                "pso_feature_set_comparison.output.detail_dir",
                default="outputs/evaluation/pso_feature_sets",
            ),
            project_root=project_root,
        ),
        "figure_dir": resolve_path(
            get_config_value(
                config,
                "pso_feature_set_comparison.output.figure_dir",
                default="outputs/evaluation/figures",
            ),
            project_root=project_root,
        ),
    }


def get_num_clusters(config: dict[str, Any], num_targets: int) -> int:
    """Resolve the PSO cluster count using target_clustering settings."""
    requested = int(get_config_value(config, "target_clustering.num_clusters", default=3))
    auto_adjust = bool(
        get_config_value(
            config,
            "target_clustering.auto_adjust_num_clusters",
            default=True,
        )
    )

    if requested <= num_targets:
        return requested
    if auto_adjust:
        return num_targets
    raise RuntimeError(
        "num_targets is smaller than num_clusters: "
        f"num_targets={num_targets}, num_clusters={requested}"
    )


def load_pso_config(config: dict[str, Any]) -> PSOClustererConfig:
    """Load PSO config from target_clustering.pso."""
    prefix = "target_clustering.pso"
    return PSOClustererConfig(
        num_particles=int(get_config_value(config, f"{prefix}.num_particles", 30)),
        max_iter=int(get_config_value(config, f"{prefix}.max_iter", 100)),
        inertia_weight=float(get_config_value(config, f"{prefix}.inertia_weight", 0.7)),
        cognitive_weight=float(
            get_config_value(config, f"{prefix}.cognitive_weight", 1.5)
        ),
        social_weight=float(get_config_value(config, f"{prefix}.social_weight", 1.5)),
        compactness_weight=float(
            get_config_value(config, f"{prefix}.compactness_weight", 1.0)
        ),
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


def plot_bar(
    rows: list[dict[str, Any]],
    metric_name: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a bar chart for one metric."""
    names = [str(row["feature_set_name"]) for row in rows]
    values = [float(row[metric_name]) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, values, color="#4C78A8")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Feature set")
    ax.set_title(metric_name)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cluster_sizes(
    cluster_counts_by_feature_set: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save grouped bar charts of cluster sizes for each feature set."""
    feature_set_names = list(cluster_counts_by_feature_set.keys())
    num_clusters = max(len(counts) for counts in cluster_counts_by_feature_set.values())
    x = np.arange(num_clusters)
    width = 0.8 / max(len(feature_set_names), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, feature_set_name in enumerate(feature_set_names):
        counts = cluster_counts_by_feature_set[feature_set_name]
        offset = (index - (len(feature_set_names) - 1) / 2) * width
        ax.bar(x + offset, counts, width=width, label=feature_set_name)

    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Target count")
    ax.set_title("cluster_size")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cluster_map_xy(
    detail_rows_by_feature_set: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Save a 2x2 x-y scatter map colored by cluster id."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_plots = len(detail_rows_by_feature_set)
    cols = 2
    rows = int(math.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows), squeeze=False)

    for axis in axes.ravel()[num_plots:]:
        axis.axis("off")

    for axis, (feature_set_name, detail_rows) in zip(
        axes.ravel(),
        detail_rows_by_feature_set.items(),
    ):
        x_values = [float(row["x"]) for row in detail_rows]
        y_values = [float(row["y"]) for row in detail_rows]
        labels = [int(row["cluster_id"]) for row in detail_rows]
        scatter = axis.scatter(
            x_values,
            y_values,
            c=labels,
            cmap="tab10",
            s=45,
            edgecolors="black",
            linewidths=0.3,
        )
        axis.set_title(feature_set_name)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(True, linestyle="--", alpha=0.3)
        legend = axis.legend(
            *scatter.legend_elements(),
            title="cluster_id",
            loc="best",
        )
        axis.add_artist(legend)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_figures(
    summary_rows: list[dict[str, Any]],
    detail_rows_by_feature_set: dict[str, list[dict[str, Any]]],
    cluster_counts_by_feature_set: dict[str, np.ndarray],
    figure_dir: Path,
) -> None:
    """Save all comparison figures."""
    plot_bar(
        summary_rows,
        metric_name="best_score",
        ylabel="Best score",
        output_path=figure_dir / "pso_feature_set_best_score.png",
    )
    plot_bar(
        summary_rows,
        metric_name="compactness",
        ylabel="Compactness",
        output_path=figure_dir / "pso_feature_set_compactness.png",
    )
    plot_bar(
        summary_rows,
        metric_name="balance_penalty",
        ylabel="Balance penalty",
        output_path=figure_dir / "pso_feature_set_balance.png",
    )
    plot_cluster_map_xy(
        detail_rows_by_feature_set,
        output_path=figure_dir / "pso_feature_set_cluster_map_xy.png",
    )
    plot_cluster_sizes(
        cluster_counts_by_feature_set,
        output_path=figure_dir / "pso_feature_set_cluster_size.png",
    )


def main() -> None:
    """Run the PSO feature-set comparison experiment."""
    args = parse_args()
    project_root = get_project_root()
    config_path = resolve_config_path(args.config)
    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    output_paths = get_output_paths(config, project_root)
    feature_sets = get_feature_sets(config)
    pso_config = load_pso_config(config)

    logger.info("=" * 80)
    logger.info("PSO feature-set comparison started.")
    logger.info("Project root: %s", project_root)
    logger.info("Config path: %s", config_path)
    logger.info("Feature sets: %s", feature_sets)
    logger.info("=" * 80)

    data = load_all_data(config)
    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    screened_set = TargetScreener(load_target_screening_config(config)).screen(
        state.targets
    )
    screened_set.validate()

    selection_result = DestroyTargetSelector(
        load_destroy_target_selection_config(config)
    ).select(screened_set)
    destroy_targets = selection_result.destroy_targets
    selected_set = selection_result.to_screened_target_set()

    if not destroy_targets:
        raise RuntimeError("DestroyTargetSelector returned no destroy targets.")

    num_clusters = get_num_clusters(config, num_targets=len(destroy_targets))
    target_values = np.asarray(
        [float(target.significance) for target in destroy_targets],
        dtype=np.float64,
    )
    target_defenses = np.asarray(
        [float(target.defense) for target in destroy_targets],
        dtype=np.float64,
    )

    summary_rows: list[dict[str, Any]] = []
    detail_rows_by_feature_set: dict[str, list[dict[str, Any]]] = {}
    cluster_counts_by_feature_set: dict[str, np.ndarray] = {}

    for feature_set_name, feature_names in feature_sets.items():
        features = build_feature_matrix(
            targets=destroy_targets,
            screened_set=selected_set,
            feature_names=feature_names,
        )
        normalized_features, _ = normalize_features(features)

        start = time.perf_counter()
        result = PSOClusterer(pso_config).cluster(
            features=normalized_features,
            num_clusters=num_clusters,
            target_values=target_values,
            target_defenses=target_defenses,
        )
        runtime_seconds = time.perf_counter() - start

        metrics = calculate_pso_metrics(
            features=normalized_features,
            labels=result.labels,
            centers=result.centers,
            num_clusters=num_clusters,
            target_values=target_values,
            target_defenses=target_defenses,
            pso_config=pso_config,
        )
        counts = np.asarray(
            [
                np.sum(result.labels == cluster_id)
                for cluster_id in range(num_clusters)
            ],
            dtype=np.int64,
        )
        cluster_counts_by_feature_set[feature_set_name] = counts

        detail_rows = build_detail_rows(
            targets=destroy_targets,
            screened_set=selected_set,
            labels=result.labels,
        )
        detail_rows_by_feature_set[feature_set_name] = detail_rows
        save_csv(
            detail_rows,
            output_paths["detail_dir"] / f"{feature_set_name}_clusters.csv",
        )

        summary_row = {
            "feature_set_name": feature_set_name,
            "feature_names": "|".join(feature_names),
            "num_targets": len(destroy_targets),
            "num_clusters": num_clusters,
            "best_score": float(result.best_score),
            "compactness": metrics["compactness"],
            "balance_penalty": metrics["balance_penalty"],
            "empty_cluster_count": int(metrics["empty_cluster_count"]),
            "cluster_size_min": int(metrics["cluster_size_min"]),
            "cluster_size_max": int(metrics["cluster_size_max"]),
            "cluster_size_std": metrics["cluster_size_std"],
            "iterations": int(result.metadata.get("num_iter", len(result.convergence_history) - 1)),
            "runtime_seconds": runtime_seconds,
        }
        summary_rows.append(summary_row)

        logger.info(
            "Feature set finished: name=%s, best_score=%.6f, "
            "compactness=%.6f, balance_penalty=%.6f, cluster_sizes=%s",
            feature_set_name,
            summary_row["best_score"],
            summary_row["compactness"],
            summary_row["balance_penalty"],
            counts.tolist(),
        )

    save_csv(summary_rows, output_paths["comparison_csv"])
    save_figures(
        summary_rows=summary_rows,
        detail_rows_by_feature_set=detail_rows_by_feature_set,
        cluster_counts_by_feature_set=cluster_counts_by_feature_set,
        figure_dir=output_paths["figure_dir"],
    )

    logger.info("Comparison CSV saved to: %s", output_paths["comparison_csv"])
    logger.info("Detail CSVs saved to: %s", output_paths["detail_dir"])
    logger.info("Figures saved to: %s", output_paths["figure_dir"])
    logger.info("PSO feature-set comparison finished.")


if __name__ == "__main__":
    main()
