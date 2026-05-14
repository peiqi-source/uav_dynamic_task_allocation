"""Random Forest 训练报告和指标文件输出。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def save_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    output_path: str | Path,
) -> Path:
    """
    保存随机森林特征重要性。

    参数：
        model: 已训练好的 RandomForestClassifier。
        feature_names: 与训练矩阵列顺序一致的特征名。
        output_path: CSV 输出路径。

    返回：
        实际写入的 CSV 路径。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_report_text(
    *,
    label_source: str,
    feature_names: list[str],
    label_counts: dict[int, int],
    use_holdout_split: bool,
    train_size: int,
    test_size: int,
    classification_report_text: str,
    confusion_matrix_value: np.ndarray,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """
    构造可落盘的训练报告文本。

    参数：
        label_source: 标签来源，支持 kmeans_bootstrap 或 label_csv。
        feature_names: 参与训练的特征名。
        label_counts: 各类别样本数量。
        use_holdout_split: 是否使用 holdout train/test split。
        train_size: 训练样本数。
        test_size: 测试样本数；全量训练时为 0。
        classification_report_text: sklearn classification_report 输出文本。
        confusion_matrix_value: sklearn confusion_matrix 输出矩阵。
        extra_metadata: 额外写入报告的键值信息。

    返回：
        完整报告文本。
    """
    metadata = extra_metadata or {}
    metadata_lines = "".join(
        f"{key}: {value}\n" for key, value in sorted(metadata.items())
    )

    return (
        "Destroy Target Random Forest Report\n"
        "===================================\n\n"
        f"label_source: {label_source}\n"
        f"feature_names: {feature_names}\n"
        f"label_counts: {label_counts}\n"
        f"use_holdout_split: {use_holdout_split}\n"
        f"train_size: {train_size}\n"
        f"test_size: {test_size}\n"
        f"{metadata_lines}\n"
        "classification_report:\n"
        f"{classification_report_text}\n\n"
        "confusion_matrix:\n"
        f"{confusion_matrix_value}\n"
    )


def save_report_text(report_text: str, output_path: str | Path) -> Path:
    """
    保存分类报告文本。

    参数：
        report_text: build_report_text 生成的报告内容。
        output_path: 文本报告输出路径。

    返回：
        实际写入的报告路径。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
    return path

