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
