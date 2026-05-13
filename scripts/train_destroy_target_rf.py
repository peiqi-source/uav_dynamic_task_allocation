"""训练destroy目标随机森林脚本，封装可直接运行的实验、检查或可视化流程。"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from uav_dynamic_task_allocation.core.entities import build_battlefield_state
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
from uav_dynamic_task_allocation.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train Random Forest model for destroy target selection."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file.",
    )

    return parser.parse_args()


def resolve_config_path(config_path: str) -> Path:
    """解析配置文件路径。"""
    path = Path(config_path)

    if path.is_absolute():
        return path

    return get_project_root() / path


def build_features_from_screened_set(
    screened_set,
    feature_names: list[str],
) -> tuple[np.ndarray, list[int], pd.DataFrame]:
    """
    从 ScreenedTargetSet 中构造 RF 训练特征。

    feature_names 由 YAML 控制，例如：
        score
        normalized_distance
        angle
        adjusted_defense
        adjusted_significance

    返回：
        x: 特征矩阵
        target_ids: 目标编号列表
        feature_df: 方便保存和检查的 DataFrame
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
            if feature_name == "score":
                value = float(target_score.score)
            elif feature_name in target_score.components:
                value = float(target_score.components[feature_name])
            elif feature_name in target_score.metadata:
                value = float(target_score.metadata[feature_name])
            else:
                raise RuntimeError(
                    f"Feature '{feature_name}' not found for target_id={target_id}. "
                    f"Available components={list(target_score.components.keys())}, "
                    f"metadata={list(target_score.metadata.keys())}"
                )

            row[feature_name] = value
            feature_values.append(value)

        rows.append(row)
        features.append(feature_values)
        target_ids.append(target_id)

    x = np.array(features, dtype=np.float64)
    feature_df = pd.DataFrame(rows)

    return x, target_ids, feature_df


def build_labels_from_kmeans_bootstrap(
    screened_set,
    config: dict[str, Any],
) -> dict[int, int]:
    """
    用 weighted_kmeans 的结果生成 RF 伪标签。

    注意：
    这是工程调试方式，不代表真实监督标签。
    正式论文实验应该使用专家标注、历史任务结果或人工构建 label_csv。
    """
    original_method = get_config_value(
        config,
        "destroy_target_selection.method",
        default="weighted_kmeans",
    )

    if "destroy_target_selection" not in config:
        config["destroy_target_selection"] = {}

    config["destroy_target_selection"]["method"] = "weighted_kmeans"

    selector_config = load_destroy_target_selection_config(config)
    selector = DestroyTargetSelector(selector_config)
    result = selector.select(screened_set)

    config["destroy_target_selection"]["method"] = original_method

    destroy_ids = {int(target.target_id) for target in result.destroy_targets}

    return {
        int(target.target_id): 1 if int(target.target_id) in destroy_ids else 0
        for target in screened_set.candidate_targets
    }


def build_labels_from_csv(
    screened_set,
    label_csv_path: str,
    target_id_column: str,
    label_column: str,
    project_root: Path,
) -> dict[int, int]:
    """
    从人工标签 CSV 中读取监督学习标签。

    CSV 至少包含：
        target_id,is_destroy_target

    is_destroy_target:
        1 表示进入摧毁目标集
        0 表示不进入摧毁目标集
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

    required_ids = {
        int(target.target_id)
        for target in screened_set.candidate_targets
    }

    missing_ids = sorted(required_ids - set(label_map.keys()))

    if missing_ids:
        raise RuntimeError(
            f"Label CSV does not contain labels for target ids: {missing_ids}"
        )

    return {
        target_id: int(label_map[target_id])
        for target_id in required_ids
    }


def should_use_holdout_split(
    y: np.ndarray,
    test_size: float,
) -> bool:
    """
    判断当前数据是否适合划分 train/test。

    当前项目 debug 数据通常只有 20 个目标，样本数较小。
    如果某一类样本太少，强行 stratify split 会报错。
    """
    if len(y) < 8:
        return False

    counts = Counter(y.tolist())

    if len(counts) < 2:
        return False

    if min(counts.values()) < 2:
        return False

    if not 0.0 < test_size < 1.0:
        return False

    return True


def save_training_data(
    feature_df: pd.DataFrame,
    target_ids: list[int],
    y: np.ndarray,
    output_path: Path,
) -> None:
    """保存 RF 训练数据，便于检查标签和特征。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_df = feature_df.copy()
    label_map = {
        int(target_id): int(label)
        for target_id, label in zip(target_ids, y, strict=False)
    }

    label_df["is_destroy_target"] = label_df["target_id"].map(label_map)
    label_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """保存特征重要性。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        无返回值；通过状态变更、文件输出或日志记录体现执行结果。
    """
    args = parse_args()

    project_root = get_project_root()
    config_path = resolve_config_path(args.config)

    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("=" * 80)
    logger.info("Destroy target RF training started.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config path: {config_path}")
    logger.info("=" * 80)

    seed = int(
        get_config_value(
            config,
            "destroy_target_rf_training.random_seed",
            default=get_config_value(config, "experiment.seed", default=42),
        )
    )
    set_seed(seed)

    data = load_all_data(config)

    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    screening_config = load_target_screening_config(config)
    screener = TargetScreener(screening_config)
    screened_set = screener.screen(state.targets)

    feature_names = list(
        get_config_value(
            config,
            "destroy_target_selection.random_forest.feature_names",
            default=[
                "score",
                "normalized_distance",
                "angle",
                "adjusted_defense",
                "adjusted_significance",
            ],
        )
    )

    x, target_ids, feature_df = build_features_from_screened_set(
        screened_set=screened_set,
        feature_names=feature_names,
    )

    label_source = str(
        get_config_value(
            config,
            "destroy_target_rf_training.label_source",
            default="kmeans_bootstrap",
        )
    )

    if label_source == "kmeans_bootstrap":
        label_map = build_labels_from_kmeans_bootstrap(
            screened_set=screened_set,
            config=config,
        )
        logger.warning(
            "Using kmeans_bootstrap labels. "
            "This is only for engineering validation, not final supervised learning."
        )

    elif label_source == "label_csv":
        label_map = build_labels_from_csv(
            screened_set=screened_set,
            label_csv_path=str(
                get_config_value(
                    config,
                    "destroy_target_rf_training.label_csv_path",
                    default="data/processed/destroy_target_labels.csv",
                )
            ),
            target_id_column=str(
                get_config_value(
                    config,
                    "destroy_target_rf_training.target_id_column",
                    default="target_id",
                )
            ),
            label_column=str(
                get_config_value(
                    config,
                    "destroy_target_rf_training.label_column",
                    default="is_destroy_target",
                )
            ),
            project_root=project_root,
        )

    else:
        raise RuntimeError(
            f"Unsupported destroy_target_rf_training.label_source: {label_source}"
        )

    y = np.array(
        [
            int(label_map[target_id])
            for target_id in target_ids
        ],
        dtype=np.int64,
    )

    label_counts = Counter(y.tolist())
    logger.info(f"Feature names: {feature_names}")
    logger.info(f"Feature matrix shape: {x.shape}")
    logger.info(f"Label counts: {dict(label_counts)}")

    if len(label_counts) < 2:
        raise RuntimeError(
            "Random Forest training requires at least two classes. "
            f"Got labels: {sorted(label_counts.keys())}"
        )

    training_data_path = resolve_path(
        str(
            get_config_value(
                config,
                "destroy_target_rf_training.output.training_data_csv",
                default="outputs/intermediate/destroy_target_rf_training_data.csv",
            )
        ),
        project_root=project_root,
    )

    save_training_data(
        feature_df=feature_df,
        target_ids=target_ids,
        y=y,
        output_path=training_data_path,
    )

    n_estimators = int(
        get_config_value(
            config,
            "destroy_target_rf_training.n_estimators",
            default=200,
        )
    )

    max_depth_raw = get_config_value(
        config,
        "destroy_target_rf_training.max_depth",
        default=None,
    )
    max_depth = None if max_depth_raw in {None, "null"} else int(max_depth_raw)

    test_size = float(
        get_config_value(
            config,
            "destroy_target_rf_training.test_size",
            default=0.25,
        )
    )

    use_holdout = should_use_holdout_split(y=y, test_size=test_size)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        class_weight="balanced",
    )

    if use_holdout:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        report = classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, y_pred)

        logger.info("Using holdout train/test split.")
        logger.info(f"Train size: {len(y_train)}, test size: {len(y_test)}")

    else:
        model.fit(x, y)
        y_pred = model.predict(x)

        report = classification_report(
            y,
            y_pred,
            digits=4,
            zero_division=0,
        )
        matrix = confusion_matrix(y, y_pred)

        logger.warning(
            "Dataset is too small or class distribution is imbalanced. "
            "Training and reporting on the full dataset. "
            "This is acceptable for engineering validation, but not for final evaluation."
        )

    logger.info("Random Forest classification report:")
    logger.info("\n" + report)
    logger.info(f"Confusion matrix:\n{matrix}")

    model_path = resolve_path(
        str(
            get_config_value(
                config,
                "destroy_target_rf_training.model_path",
                default="checkpoints/random_forest/destroy_target_rf.joblib",
            )
        ),
        project_root=project_root,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    report_path = resolve_path(
        str(
            get_config_value(
                config,
                "destroy_target_rf_training.output.report_path",
                default="outputs/evaluation/destroy_target_rf_report.txt",
            )
        ),
        project_root=project_root,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_text = (
        "Destroy Target Random Forest Report\n"
        "===================================\n\n"
        f"label_source: {label_source}\n"
        f"feature_names: {feature_names}\n"
        f"label_counts: {dict(label_counts)}\n"
        f"use_holdout_split: {use_holdout}\n\n"
        "classification_report:\n"
        f"{report}\n\n"
        "confusion_matrix:\n"
        f"{matrix}\n"
    )
    report_path.write_text(report_text, encoding="utf-8")

    importance_path = resolve_path(
        str(
            get_config_value(
                config,
                "destroy_target_rf_training.output.feature_importance_csv",
                default="outputs/evaluation/destroy_target_rf_feature_importance.csv",
            )
        ),
        project_root=project_root,
    )

    save_feature_importance(
        model=model,
        feature_names=feature_names,
        output_path=importance_path,
    )

    logger.info(f"RF model saved to: {model_path}")
    logger.info(f"RF report saved to: {report_path}")
    logger.info(f"RF feature importance saved to: {importance_path}")
    logger.info(f"RF training data saved to: {training_data_path}")
    logger.info("Destroy target RF training finished successfully.")


if __name__ == "__main__":
    main()