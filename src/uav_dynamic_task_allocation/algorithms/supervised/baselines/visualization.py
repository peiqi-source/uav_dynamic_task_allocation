"""监督学习 baseline 对比可视化。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from uav_dynamic_task_allocation.algorithms.supervised.baselines.trainer import (
    SupervisedBaselineResult,
)
from uav_dynamic_task_allocation.algorithms.supervised.random_forest.visualization import (
    plot_confusion_matrix,
)


def plot_baseline_confusion_matrix(
    matrix,
    output_path: str | Path,
) -> Path:
    """保存 baseline 二分类混淆矩阵图。"""
    return plot_confusion_matrix(matrix=matrix, output_path=output_path)


def plot_classifier_comparison_metrics(
    results: list[SupervisedBaselineResult],
    output_path: str | Path,
) -> Path:
    """绘制 Decision Tree、KNN、RF 的主要指标 grouped bar chart。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    model_names = [result.model_name for result in results]
    values = np.array(
        [
            [
                result.accuracy,
                result.precision_macro,
                result.recall_macro,
                result.f1_macro,
            ]
            for result in results
        ],
        dtype=float,
    )

    x_positions = np.arange(len(metrics))
    width = 0.8 / max(1, len(model_names))

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_index, model_name in enumerate(model_names):
        offset = (model_index - (len(model_names) - 1) / 2) * width
        ax.bar(
            x_positions + offset,
            values[model_index],
            width,
            label=model_name,
        )

    ax.set_xticks(x_positions, labels=metrics)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Destroy Target Classifier Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

