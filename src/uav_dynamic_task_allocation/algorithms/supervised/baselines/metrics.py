"""监督学习 baseline 对比指标输出。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from uav_dynamic_task_allocation.algorithms.supervised.baselines.trainer import (
    SupervisedBaselineResult,
)


def save_comparison_csv(
    results: list[SupervisedBaselineResult],
    output_path: str | Path,
) -> Path:
    """保存多模型指标对比 CSV。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model_name": result.model_name,
            "accuracy": result.accuracy,
            "precision_macro": result.precision_macro,
            "recall_macro": result.recall_macro,
            "f1_macro": result.f1_macro,
            "precision_weighted": result.precision_weighted,
            "recall_weighted": result.recall_weighted,
            "f1_weighted": result.f1_weighted,
            "train_size": result.train_size,
            "test_size": result.test_size,
            "label_source": result.label_source,
            "feature_names": "|".join(result.feature_names),
        }
        for result in results
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_model_report_text(result: SupervisedBaselineResult) -> str:
    """构造单个模型的评估报告文本。"""
    return (
        f"{result.model_name} Destroy Target Classification Report\n"
        "================================================\n\n"
        f"label_source: {result.label_source}\n"
        f"feature_names: {result.feature_names}\n"
        f"use_holdout_split: {result.use_holdout_split}\n"
        f"train_size: {result.train_size}\n"
        f"test_size: {result.test_size}\n"
        f"accuracy: {result.accuracy:.6f}\n"
        f"precision_macro: {result.precision_macro:.6f}\n"
        f"recall_macro: {result.recall_macro:.6f}\n"
        f"f1_macro: {result.f1_macro:.6f}\n"
        f"precision_weighted: {result.precision_weighted:.6f}\n"
        f"recall_weighted: {result.recall_weighted:.6f}\n"
        f"f1_weighted: {result.f1_weighted:.6f}\n\n"
        "classification_report:\n"
        f"{result.classification_report}\n\n"
        "confusion_matrix:\n"
        f"{result.confusion_matrix}\n"
    )


def save_model_report(
    result: SupervisedBaselineResult,
    output_path: str | Path,
) -> Path:
    """保存单个模型评估报告。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_model_report_text(result), encoding="utf-8")
    return path

