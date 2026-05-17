"""Matplotlib visualizations for PPO target regrouping."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_rows(metrics_csv: str | Path) -> list[dict[str, Any]]:
    with Path(metrics_csv).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _float_column(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        values.append(float(raw) if raw not in {"", None} else 0.0)
    return values


def _save_line(
    rows: list[dict[str, Any]],
    columns: list[str],
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = _float_column(rows, "episode")
    fig, ax = plt.subplots(figsize=(10, 5))
    for column in columns:
        ax.plot(x_values, _float_column(rows, column), label=column)
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training_reward(metrics_csv: str | Path, output_path: str | Path) -> None:
    rows = _read_rows(metrics_csv)
    _save_line(rows, ["total_reward"], output_path, "PPO training reward", "reward")


def plot_training_score(metrics_csv: str | Path, output_path: str | Path) -> None:
    rows = _read_rows(metrics_csv)
    _save_line(rows, ["final_score"], output_path, "PPO training score", "score")


def plot_training_losses(metrics_csv: str | Path, output_path: str | Path) -> None:
    rows = _read_rows(metrics_csv)
    _save_line(
        rows,
        ["policy_loss", "value_loss", "entropy"],
        output_path,
        "PPO training losses",
        "value",
    )


def plot_training_reward_smoothed(
    metrics_csv: str | Path,
    output_path: str | Path,
) -> None:
    rows = _read_rows(metrics_csv)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = _float_column(rows, "episode")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_values, _float_column(rows, "total_reward"), alpha=0.25, label="total_reward")
    ax.plot(x_values, _float_column(rows, "moving_avg_reward_20"), label="moving_avg_reward_20")
    ax.plot(x_values, _float_column(rows, "moving_avg_reward_50"), label="moving_avg_reward_50")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title("PPO training reward (smoothed)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training_score_smoothed(
    metrics_csv: str | Path,
    output_path: str | Path,
) -> None:
    rows = _read_rows(metrics_csv)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = _float_column(rows, "episode")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_values, _float_column(rows, "final_score"), alpha=0.25, label="final_score")
    ax.plot(x_values, _float_column(rows, "moving_avg_score_20"), label="moving_avg_score_20")
    ax.plot(x_values, _float_column(rows, "moving_avg_score_50"), label="moving_avg_score_50")
    ax.set_xlabel("episode")
    ax.set_ylabel("score")
    ax.set_title("PPO training score (smoothed)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_event_type_reward_boxplot(
    metrics_csv: str | Path,
    output_path: str | Path,
) -> None:
    _plot_event_boxplot(
        metrics_csv=metrics_csv,
        output_path=output_path,
        value_key="total_reward",
        title="Reward by event type",
        ylabel="reward",
    )


def plot_event_type_score_boxplot(
    metrics_csv: str | Path,
    output_path: str | Path,
) -> None:
    _plot_event_boxplot(
        metrics_csv=metrics_csv,
        output_path=output_path,
        value_key="final_score",
        title="Final score by event type",
        ylabel="score",
    )


def plot_ppo_vs_pso_metrics(comparison_csv: str | Path, output_path: str | Path) -> None:
    rows = _read_rows(comparison_csv)
    methods = [row["method"] for row in rows]
    metric_names = [
        "final_score",
        "compactness",
        "count_balance",
        "workload_balance",
        "empty_cluster_count",
    ]
    x = np.arange(len(metric_names))
    width = 0.8 / max(len(methods), 1)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for index, row in enumerate(rows):
        offset = (index - (len(rows) - 1) / 2) * width
        values = [float(row.get(metric, 0.0) or 0.0) for metric in metric_names]
        ax.bar(x + offset, values, width=width, label=row["method"])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20)
    ax.set_ylabel("metric value")
    ax.set_title("PPO vs PSO clustering metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_event_boxplot(
    metrics_csv: str | Path,
    output_path: str | Path,
    value_key: str,
    title: str,
    ylabel: str,
) -> None:
    rows = _read_rows(metrics_csv)
    values_by_event: dict[str, list[float]] = {}
    for row in rows:
        event_type = row.get("event_type", "unknown") or "unknown"
        raw_value = row.get(value_key, "")
        if raw_value in {"", None}:
            continue
        values_by_event.setdefault(event_type, []).append(float(raw_value))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = sorted(values_by_event)
    data = [values_by_event[label] for label in labels]
    if data:
        ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_xlabel("event_type")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_cluster_map(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        features[:, 0],
        features[:, 1],
        c=labels,
        cmap="tab10",
        s=45,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    legend = ax.legend(*scatter.legend_elements(), title="cluster_id", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_dynamic_added_targets_before_after(
    xy_features,
    before_labels,
    after_labels,
    is_new_target,
    output_path,
    title_before: str = "Before PPO regrouping",
    title_after: str = "After PPO regrouping",
) -> None:
    """Plot before/after PPO regrouping with newly added targets highlighted."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xy_features = np.asarray(xy_features, dtype=np.float64)
    before_labels = np.asarray(before_labels, dtype=np.int64)
    after_labels = np.asarray(after_labels, dtype=np.int64)
    is_new_target = np.asarray(is_new_target, dtype=bool)

    x_min, x_max = float(np.min(xy_features[:, 0])), float(np.max(xy_features[:, 0]))
    y_min, y_max = float(np.min(xy_features[:, 1])), float(np.max(xy_features[:, 1]))
    x_margin = max((x_max - x_min) * 0.05, 1.0)
    y_margin = max((y_max - y_min) * 0.05, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for axis, labels, title in [
        (axes[0], before_labels, title_before),
        (axes[1], after_labels, title_after),
    ]:
        normal_mask = ~is_new_target
        scatter = axis.scatter(
            xy_features[normal_mask, 0],
            xy_features[normal_mask, 1],
            c=labels[normal_mask],
            cmap="tab10",
            marker="o",
            s=45,
            edgecolors="black",
            linewidths=0.3,
        )
        axis.scatter(
            xy_features[is_new_target, 0],
            xy_features[is_new_target, 1],
            c="black",
            marker="^",
            s=120,
            edgecolors="black",
            linewidths=1.8,
            label="New target",
            zorder=5,
        )
        if normal_mask.any():
            legend = axis.legend(
                *scatter.legend_elements(),
                title="cluster_id",
                loc="best",
            )
            axis.add_artist(legend)
        axis.legend(loc="upper right")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_xlim(x_min - x_margin, x_max + x_margin)
        axis.set_ylim(y_min - y_margin, y_max + y_margin)
        axis.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
