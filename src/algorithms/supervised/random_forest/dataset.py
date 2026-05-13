"""Random Forest 摧毁目标集训练数据构造。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from uav_dynamic_task_allocation.core.contracts import ScreenedTargetSet
from uav_dynamic_task_allocation.preprocessing.destroy_target_selection import (
    DestroyTargetSelector,
    load_destroy_target_selection_config,
)
from uav_dynamic_task_allocation.utils.config import get_config_value, resolve_path


KMEANS_BOOTSTRAP_WARNING = (
    "kmeans_bootstrap 标签只是工程伪标签，用于验证训练链路，不是正式监督学习标签。"
    "正式实验应使用专家标注、历史任务结果或人工构建的 label_csv。"
)


@dataclass(frozen=True)
class RandomForestDataset:
    """
    Random Forest 训练数据集。

    属性：
        x: 特征矩阵，形状为 [num_targets, num_features]。
        y: 二分类标签，1 表示进入摧毁目标集，0 表示不进入。
        target_ids: 与 x/y 行顺序一一对应的目标编号。
        feature_df: 包含目标基础字段和特征列的 DataFrame，便于导出检查。
        label_source: 标签来源，支持 kmeans_bootstrap 或 label_csv。
    """

    x: np.ndarray
    y: np.ndarray
    target_ids: list[int]
    feature_df: pd.DataFrame
    label_source: str


def build_features_from_screened_set(
    screened_set: ScreenedTargetSet,
    feature_names: list[str],
) -> tuple[np.ndarray, list[int], pd.DataFrame]:
    """
    从 ScreenedTargetSet 构造 Random Forest 训练特征。

    参数：
        screened_set: TargetScreener.screen 输出的目标筛选结果。
        feature_names: 需要从 TargetScore 中读取的特征名，顺序即模型输入列顺序。

    返回：
        x: 特征矩阵。
        target_ids: 与特征矩阵行顺序一致的目标编号列表。
        feature_df: 便于保存和人工检查的特征明细表。
    """
    rows: list[dict[str, Any]] = []
    features: list[list[float]] = []
    target_ids: list[int] = []

    for target in screened_set.candidate_targets:
        target_id = int(target.target_id)
        target_score = screened_set.target_scores[target_id]

        row: dict[str, Any] = {
            "target_id": target_id,
            "target_type": int(
                target.target_type.value
                if hasattr(target.target_type, "value")
                else target.target_type
            ),
            "x": float(target.position.x),
            "y": float(target.position.y),
            "defense": float(target.defense),
            "significance": float(target.significance),
        }
        feature_values: list[float] = []

        for feature_name in feature_names:
            value = _get_feature_value(
                screened_set=screened_set,
                target_id=target_id,
                feature_name=feature_name,
            )
            row[feature_name] = value
            feature_values.append(value)

        rows.append(row)
        features.append(feature_values)
        target_ids.append(target_id)

    return np.array(features, dtype=np.float64), target_ids, pd.DataFrame(rows)


def build_labels_from_kmeans_bootstrap(
    screened_set: ScreenedTargetSet,
    config: dict[str, Any],
) -> dict[int, int]:
    """
    使用 weighted_kmeans 结果生成 Random Forest 工程伪标签。

    参数：
        screened_set: 目标筛选结果。
        config: 项目总配置；函数内部会复制配置，不会修改调用方对象。

    返回：
        target_id 到二分类标签的映射，1 表示摧毁目标，0 表示非摧毁目标。

    警告：
        kmeans_bootstrap 只是工程伪标签，不是正式监督学习标签。
    """
    warnings.warn(KMEANS_BOOTSTRAP_WARNING, UserWarning, stacklevel=2)

    local_config = deepcopy(config)
    if "destroy_target_selection" not in local_config:
        local_config["destroy_target_selection"] = {}
    local_config["destroy_target_selection"]["method"] = "weighted_kmeans"

    selector_config = load_destroy_target_selection_config(local_config)
    result = DestroyTargetSelector(selector_config).select(screened_set)
    destroy_ids = {int(target.target_id) for target in result.destroy_targets}

    return {
        int(target.target_id): 1 if int(target.target_id) in destroy_ids else 0
        for target in screened_set.candidate_targets
    }


def build_labels_from_csv(
    screened_set: ScreenedTargetSet,
    label_csv_path: str | Path,
    target_id_column: str,
    label_column: str,
    project_root: str | Path | None = None,
) -> dict[int, int]:
    """
    从人工标签 CSV 读取监督学习标签。

    参数：
        screened_set: 目标筛选结果。
        label_csv_path: 标签 CSV 路径。
        target_id_column: 目标编号列名。
        label_column: 标签列名，1 表示摧毁目标，0 表示非摧毁目标。
        project_root: 解析相对路径时使用的项目根目录。

    返回：
        target_id 到二分类标签的映射。
    """
    path = resolve_path(label_csv_path, project_root=project_root)
    if not path.exists():
        raise FileNotFoundError(f"Label CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    if target_id_column not in df.columns:
        raise RuntimeError(f"Missing target id column: {target_id_column}")
    if label_column not in df.columns:
        raise RuntimeError(f"Missing label column: {label_column}")

    label_map = {
        int(row[target_id_column]): int(row[label_column])
        for _, row in df.iterrows()
    }
    required_ids = {int(target.target_id) for target in screened_set.candidate_targets}
    missing_ids = sorted(required_ids - set(label_map.keys()))
    if missing_ids:
        raise RuntimeError(
            f"Label CSV does not contain labels for target ids: {missing_ids}"
        )

    return {target_id: int(label_map[target_id]) for target_id in required_ids}


def build_destroy_target_rf_dataset(
    *,
    screened_set: ScreenedTargetSet,
    feature_names: list[str],
    label_source: str,
    config: dict[str, Any],
    project_root: str | Path | None = None,
    label_csv_path: str | Path | None = None,
    target_id_column: str = "target_id",
    label_column: str = "is_destroy_target",
) -> RandomForestDataset:
    """
    构造完整 Random Forest 训练数据集。

    参数：
        screened_set: 目标筛选结果。
        feature_names: 模型输入特征名。
        label_source: 标签来源，支持 kmeans_bootstrap 和 label_csv。
        config: 项目总配置，kmeans_bootstrap 伪标签会读取其中的 KMeans 配置。
        project_root: 解析相对路径时使用的项目根目录。
        label_csv_path: label_csv 模式下的标签文件路径。
        target_id_column: label_csv 中目标编号列名。
        label_column: label_csv 中标签列名。

    返回：
        RandomForestDataset，包含 X、y、target_ids 和 feature_df。
    """
    x, target_ids, feature_df = build_features_from_screened_set(
        screened_set=screened_set,
        feature_names=feature_names,
    )

    if label_source == "kmeans_bootstrap":
        label_map = build_labels_from_kmeans_bootstrap(screened_set, config)
    elif label_source == "label_csv":
        if label_csv_path is None:
            label_csv_path = get_config_value(
                config,
                "destroy_target_rf_training.label_csv_path",
                default="data/processed/destroy_target_labels.csv",
            )
        label_map = build_labels_from_csv(
            screened_set=screened_set,
            label_csv_path=label_csv_path,
            target_id_column=target_id_column,
            label_column=label_column,
            project_root=project_root,
        )
    else:
        raise RuntimeError(
            f"Unsupported destroy_target_rf_training.label_source: {label_source}"
        )

    y = np.array([int(label_map[target_id]) for target_id in target_ids], dtype=np.int64)
    return RandomForestDataset(
        x=x,
        y=y,
        target_ids=target_ids,
        feature_df=feature_df,
        label_source=label_source,
    )


def save_training_data(
    feature_df: pd.DataFrame,
    target_ids: list[int],
    y: np.ndarray,
    output_path: str | Path,
) -> Path:
    """
    保存 Random Forest 训练数据。

    参数：
        feature_df: build_features_from_screened_set 输出的特征明细。
        target_ids: 与 y 行顺序一致的目标编号。
        y: 二分类标签数组。
        output_path: CSV 输出路径。

    返回：
        实际写入的训练数据 CSV 路径。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    label_df = feature_df.copy()
    label_map = {
        int(target_id): int(label)
        for target_id, label in zip(target_ids, y, strict=False)
    }
    label_df["is_destroy_target"] = label_df["target_id"].map(label_map)
    label_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _get_feature_value(
    screened_set: ScreenedTargetSet,
    target_id: int,
    feature_name: str,
) -> float:
    """从 TargetScore 中读取指定 RF 特征值。"""
    target_score = screened_set.target_scores[target_id]

    if feature_name == "score":
        return float(target_score.score)
    if feature_name in target_score.components:
        return float(target_score.components[feature_name])
    if feature_name in target_score.metadata:
        return float(target_score.metadata[feature_name])

    raise RuntimeError(
        f"Feature '{feature_name}' not found for target_id={target_id}. "
        f"Available components={list(target_score.components.keys())}, "
        f"metadata={list(target_score.metadata.keys())}"
    )

