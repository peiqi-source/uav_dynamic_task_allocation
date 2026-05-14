"""Random Forest 摧毁目标集决策完整训练流水线。"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

from uav_dynamic_task_allocation.algorithms.supervised.random_forest.checkpoint import (
    save_random_forest_model,
)
from uav_dynamic_task_allocation.algorithms.supervised.random_forest.dataset import (
    build_destroy_target_rf_dataset,
    save_training_data,
)
from uav_dynamic_task_allocation.algorithms.supervised.random_forest.metrics import (
    build_report_text,
    save_report_text,
)
from uav_dynamic_task_allocation.algorithms.supervised.random_forest.trainer import (
    RandomForestDestroyTargetTrainer,
)
from uav_dynamic_task_allocation.algorithms.supervised.random_forest.visualization import (
    plot_confusion_matrix,
    plot_destroy_target_3d_value_map,
    plot_destroy_target_map,
    plot_feature_importance,
    plot_score_distribution,
)
from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
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


def run_destroy_target_rf_training(config_path: str | Path) -> None:
    """
    运行摧毁目标集 Random Forest 训练流水线。

    参数：
        config_path: YAML 配置文件路径；相对路径会按项目根目录解析。

    返回：
        无返回值；模型、报告、特征重要性和训练数据会写入配置指定路径。
    """
    project_root = get_project_root()
    resolved_config_path = _resolve_config_path(config_path, project_root)

    config = load_and_validate_config(resolved_config_path)
    logger = setup_logger_from_config(config)

    logger.info("=" * 80)
    logger.info("Destroy target RF training started.")
    logger.info("Project root: %s", project_root)
    logger.info("Config path: %s", resolved_config_path)
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
    screened_set = TargetScreener(screening_config).screen(state.targets)

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
    label_source = str(
        get_config_value(
            config,
            "destroy_target_rf_training.label_source",
            default="kmeans_bootstrap",
        )
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        dataset = build_destroy_target_rf_dataset(
            screened_set=screened_set,
            feature_names=feature_names,
            label_source=label_source,
            config=config,
            project_root=project_root,
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
        )

    for warning_item in caught_warnings:
        logger.warning(str(warning_item.message))

    logger.info("Feature names: %s", feature_names)
    logger.info("Feature matrix shape: %s", dataset.x.shape)
    logger.info("Label counts: %s", _label_counts(dataset.y))

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
        feature_df=dataset.feature_df,
        target_ids=dataset.target_ids,
        y=dataset.y,
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
    confusion_matrix_figure_path = resolve_path(
        str(
            get_config_value(
                config,
                "destroy_target_rf_training.output.confusion_matrix_figure",
                default="outputs/evaluation/figures/rf_confusion_matrix.png",
            )
        ),
        project_root=project_root,
    )
    feature_importance_figure_path = resolve_path(
        str(
            get_config_value(
                config,
                "destroy_target_rf_training.output.feature_importance_figure",
                default="outputs/evaluation/figures/rf_feature_importance.png",
            )
        ),
        project_root=project_root,
    )
    destroy_target_map_figure_path = resolve_path(
        str(
            get_config_value(
                config,
                "destroy_target_rf_training.output.destroy_target_map_figure",
                default="outputs/evaluation/figures/rf_destroy_target_map.png",
            )
        ),
        project_root=project_root,
    )
    score_distribution_figure_path = resolve_path(
        str(
            get_config_value(
                config,
                "destroy_target_rf_training.output.score_distribution_figure",
                default="outputs/evaluation/figures/rf_score_distribution.png",
            )
        ),
        project_root=project_root,
    )
    destroy_target_3d_value_map_figure_path = resolve_path(
        str(
            get_config_value(
                config,
                "destroy_target_rf_training.output.destroy_target_3d_value_map_figure",
                default="outputs/evaluation/figures/rf_destroy_target_3d_value_map.png",
            )
        ),
        project_root=project_root,
    )

    trainer = RandomForestDestroyTargetTrainer(
        n_estimators=n_estimators,
        max_depth=max_depth,
        test_size=test_size,
        random_seed=seed,
        logger=logger,
    )
    result = trainer.train(
        x=dataset.x,
        y=dataset.y,
        feature_names=feature_names,
        feature_importance_path=importance_path,
    )

    logger.info("Random Forest classification report:")
    logger.info("\n%s", result.classification_report)
    logger.info("Confusion matrix:\n%s", result.confusion_matrix)

    save_random_forest_model(result.model, model_path)
    report_text = build_report_text(
        label_source=label_source,
        feature_names=feature_names,
        label_counts=result.label_counts,
        use_holdout_split=result.use_holdout_split,
        train_size=result.train_size,
        test_size=result.test_size,
        classification_report_text=result.classification_report,
        confusion_matrix_value=result.confusion_matrix,
        extra_metadata=result.metadata,
    )
    save_report_text(report_text, report_path)

    training_df = dataset.feature_df.copy()
    label_map = {
        int(target_id): int(label)
        for target_id, label in zip(dataset.target_ids, dataset.y, strict=False)
    }
    training_df["is_destroy_target"] = training_df["target_id"].map(label_map)

    plot_confusion_matrix(
        matrix=result.confusion_matrix,
        output_path=confusion_matrix_figure_path,
    )
    plot_feature_importance(
        feature_names=feature_names,
        importances=result.model.feature_importances_,
        output_path=feature_importance_figure_path,
    )
    plot_destroy_target_map(
        training_df=training_df,
        output_path=destroy_target_map_figure_path,
        base_position=_get_base_position(config),
    )
    plot_score_distribution(
        training_df=training_df,
        output_path=score_distribution_figure_path,
    )
    plot_destroy_target_3d_value_map(
        training_df=training_df,
        output_path=destroy_target_3d_value_map_figure_path,
        value_column="score",
    )

    logger.info("RF model saved to: %s", model_path)
    logger.info("RF report saved to: %s", report_path)
    logger.info("RF feature importance saved to: %s", importance_path)
    logger.info("RF training data saved to: %s", training_data_path)
    logger.info("RF confusion matrix figure saved to: %s", confusion_matrix_figure_path)
    logger.info(
        "RF feature importance figure saved to: %s",
        feature_importance_figure_path,
    )
    logger.info("RF destroy target map saved to: %s", destroy_target_map_figure_path)
    logger.info(
        "RF score distribution figure saved to: %s",
        score_distribution_figure_path,
    )
    logger.info(
        "RF 3D destroy target value map saved to: %s",
        destroy_target_3d_value_map_figure_path,
    )
    logger.info("Destroy target RF training finished successfully.")


def main() -> None:
    """命令行入口，便于通过 python -m 调用训练流水线。"""
    parser = argparse.ArgumentParser(
        description="Train Random Forest model for destroy target selection."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file.",
    )
    args = parser.parse_args()
    run_destroy_target_rf_training(args.config)


def _resolve_config_path(config_path: str | Path, project_root: Path) -> Path:
    """解析配置文件路径。"""
    path = Path(config_path)
    if path.is_absolute():
        return path
    return project_root / path


def _label_counts(y) -> dict[int, int]:
    """统计标签样本数量，避免在流水线中暴露 Counter 细节。"""
    counts: dict[int, int] = {}
    for value in y.tolist():
        label = int(value)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _get_base_position(config: dict) -> tuple[float, float]:
    """从配置读取基地坐标，默认使用 (0, -16000)。"""
    base_position = get_config_value(
        config,
        "resource_allocation.base_position",
        default=None,
    )
    if base_position is not None and len(base_position) >= 2:
        return float(base_position[0]), float(base_position[1])

    return (
        float(get_config_value(config, "env.base_x", default=0.0)),
        float(get_config_value(config, "env.base_y", default=-16000.0)),
    )


if __name__ == "__main__":
    main()
