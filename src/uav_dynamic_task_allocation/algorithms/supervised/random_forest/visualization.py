"""Random Forest 摧毁目标集训练可视化。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_confusion_matrix(matrix, output_path: str | Path) -> Path:
    """
    绘制并保存二分类混淆矩阵图。

    参数：
        matrix: sklearn.metrics.confusion_matrix 返回的矩阵。
        output_path: 图片保存路径。

    返回：
        实际写入的图片路径。
    """
    path = _prepare_output_path(output_path)
    display_matrix = _ensure_2x2_matrix(matrix)
    x_labels = ["Predicted Non-destroy", "Predicted Destroy"]
    y_labels = ["True Non-destroy", "True Destroy"]

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(display_matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Random Forest Confusion Matrix")

    threshold = float(display_matrix.max()) / 2.0 if display_matrix.size else 0.0
    for row_index in range(display_matrix.shape[0]):
        for col_index in range(display_matrix.shape[1]):
            value = int(display_matrix[row_index, col_index])
            text_color = "white" if value > threshold else "black"
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(
    feature_names: list[str],
    importances,
    output_path: str | Path,
) -> Path:
    """
    绘制并保存特征重要性柱状图。

    参数：
        feature_names: 特征名称列表。
        importances: RandomForestClassifier.feature_importances_。
        output_path: 图片保存路径。

    返回：
        实际写入的图片路径。
    """
    path = _prepare_output_path(output_path)
    importance_values = np.asarray(importances, dtype=float)
    order = np.argsort(importance_values)[::-1]
    sorted_features = [feature_names[index] for index in order]
    sorted_importances = importance_values[order]

    fig_height = max(4.0, 0.45 * len(sorted_features) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(sorted_features, sorted_importances, color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")

    for index, value in enumerate(sorted_importances):
        ax.text(value, index, f" {value:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_destroy_target_map(
    training_df: pd.DataFrame,
    output_path: str | Path,
    base_position: tuple[float, float] = (0.0, -16000.0),
) -> Path:
    """
    绘制并保存摧毁目标二分类空间分布图。

    参数：
        training_df: destroy_target_rf_training_data.csv 对应的 DataFrame。
        output_path: 图片保存路径。
        base_position: 基地坐标，默认 (0, -16000)。

    返回：
        实际写入的图片路径。
    """
    path = _prepare_output_path(output_path)
    _validate_training_columns(training_df, ["x", "y", "is_destroy_target"])

    destroy_df = training_df[training_df["is_destroy_target"].astype(int) == 1]
    non_destroy_df = training_df[training_df["is_destroy_target"].astype(int) == 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        non_destroy_df["x"],
        non_destroy_df["y"],
        s=28,
        marker="o",
        color="#4C78A8",
        alpha=0.75,
        label="Non-destroy target",
    )
    ax.scatter(
        destroy_df["x"],
        destroy_df["y"],
        s=40,
        marker="^",
        color="#E45756",
        alpha=0.85,
        label="Destroy target",
    )
    ax.scatter(
        [base_position[0]],
        [base_position[1]],
        s=120,
        marker="*",
        color="#222222",
        label="Base",
        zorder=5,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Destroy Target Spatial Distribution")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_destroy_target_3d_value_map(
    training_df: pd.DataFrame,
    output_path: str | Path,
    value_column: str = "score",
    cut_value: float | None = None,
) -> Path:
    """
    绘制战场目标三维价值分布图，不显示基地。

    x/y 表示目标空间坐标，z 表示目标价值属性；半透明切面用于展示
    摧毁目标集与非摧毁目标集在价值维度上的分界参考。
    """
    path = _prepare_output_path(output_path)
    _validate_training_columns(training_df, ["x", "y", value_column, "is_destroy_target"])

    plot_df = training_df.copy()
    plot_df[value_column] = plot_df[value_column].astype(float)
    destroy_df = plot_df[plot_df["is_destroy_target"].astype(int) == 1]
    non_destroy_df = plot_df[plot_df["is_destroy_target"].astype(int) == 0]

    if cut_value is None:
        cut_value = _infer_value_cut_plane(destroy_df, non_destroy_df, value_column)

    fig = plt.figure(figsize=(10, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        non_destroy_df["x"],
        non_destroy_df["y"],
        non_destroy_df[value_column],
        s=18,
        marker="s",
        color="#8F8F8F",
        alpha=0.72,
        label="Non-destroy target",
        depthshade=False,
    )
    ax.scatter(
        destroy_df["x"],
        destroy_df["y"],
        destroy_df[value_column],
        s=28,
        marker="d",
        color="#E45756",
        alpha=0.9,
        label="Destroy target",
        depthshade=False,
    )

    x_grid, y_grid = np.meshgrid(
        np.linspace(float(plot_df["x"].min()), float(plot_df["x"].max()), 2),
        np.linspace(float(plot_df["y"].min()), float(plot_df["y"].max()), 2),
    )
    z_grid = np.full_like(x_grid, float(cut_value), dtype=float)
    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color="#4C78D8",
        alpha=0.28,
        linewidth=0,
        shade=False,
    )

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_zlabel("Target value")
    ax.set_title("3D Distribution of Destroy Target Classification")
    ax.view_init(elev=14, azim=-68)
    ax.legend(loc="upper right")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_score_distribution(
    training_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    绘制并保存两类目标的 score 分布直方图。

    参数：
        training_df: destroy_target_rf_training_data.csv 对应的 DataFrame。
        output_path: 图片保存路径。

    返回：
        实际写入的图片路径。
    """
    path = _prepare_output_path(output_path)
    _validate_training_columns(training_df, ["score", "is_destroy_target"])

    destroy_scores = training_df.loc[
        training_df["is_destroy_target"].astype(int) == 1,
        "score",
    ]
    non_destroy_scores = training_df.loc[
        training_df["is_destroy_target"].astype(int) == 0,
        "score",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        non_destroy_scores,
        bins=20,
        alpha=0.7,
        color="#4C78A8",
        label="Non-destroy target",
    )
    ax.hist(
        destroy_scores,
        bins=20,
        alpha=0.7,
        color="#E45756",
        label="Destroy target",
    )
    ax.set_xlabel("score")
    ax.set_ylabel("Count")
    ax.set_title("Random Forest Training Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _prepare_output_path(output_path: str | Path) -> Path:
    """创建图片父目录并返回 Path。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_2x2_matrix(matrix) -> np.ndarray:
    """将输入矩阵整理为 2x2，便于固定二分类坐标轴显示。"""
    array = np.asarray(matrix, dtype=int)
    if array.shape == (2, 2):
        return array

    display_matrix = np.zeros((2, 2), dtype=int)
    rows = min(2, array.shape[0])
    cols = min(2, array.shape[1]) if array.ndim > 1 else 1
    if array.ndim == 1:
        display_matrix[:rows, 0] = array[:rows]
    else:
        display_matrix[:rows, :cols] = array[:rows, :cols]
    return display_matrix


def _validate_training_columns(
    training_df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """检查训练数据是否包含绘图所需列。"""
    missing_columns = [
        column for column in required_columns if column not in training_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns for RF visualization: {missing_columns}")


def _infer_value_cut_plane(
    destroy_df: pd.DataFrame,
    non_destroy_df: pd.DataFrame,
    value_column: str,
) -> float:
    """根据两类样本均值推断 3D 图中的价值切面位置。"""
    if destroy_df.empty or non_destroy_df.empty:
        merged = pd.concat([destroy_df, non_destroy_df], ignore_index=True)
        return float(merged[value_column].median())

    destroy_mean = float(destroy_df[value_column].mean())
    non_destroy_mean = float(non_destroy_df[value_column].mean())
    return (destroy_mean + non_destroy_mean) / 2.0
