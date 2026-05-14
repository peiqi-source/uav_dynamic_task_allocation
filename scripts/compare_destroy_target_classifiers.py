"""对比 RF、Decision Tree 和 KNN 摧毁目标集分类效果。"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from uav_dynamic_task_allocation.algorithms.supervised.baselines.metrics import (  # noqa: E402
    save_comparison_csv,
    save_model_report,
)
from uav_dynamic_task_allocation.algorithms.supervised.baselines.trainer import (  # noqa: E402
    SupervisedBaselineResult,
    SupervisedBaselineTrainer,
)
from uav_dynamic_task_allocation.algorithms.supervised.baselines.visualization import (  # noqa: E402
    plot_baseline_confusion_matrix,
    plot_classifier_comparison_metrics,
)
from uav_dynamic_task_allocation.algorithms.supervised.random_forest.checkpoint import (  # noqa: E402
    load_random_forest_model,
)
from uav_dynamic_task_allocation.algorithms.supervised.random_forest.dataset import (  # noqa: E402
    build_destroy_target_rf_dataset,
)
from uav_dynamic_task_allocation.core.entities import build_battlefield_state  # noqa: E402
from uav_dynamic_task_allocation.data.loaders import load_all_data  # noqa: E402
from uav_dynamic_task_allocation.preprocessing.target_screening import (  # noqa: E402
    TargetScreener,
    load_target_screening_config,
)
from uav_dynamic_task_allocation.utils.config import (  # noqa: E402
    get_config_value,
    get_project_root,
    load_and_validate_config,
    resolve_path,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config  # noqa: E402
from uav_dynamic_task_allocation.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Compare destroy-target classifiers: RF, Decision Tree, KNN."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    """运行摧毁目标集分类器对比实验。"""
    args = parse_args()
    project_root = get_project_root()
    config_path = _resolve_config_path(args.config, project_root)
    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("=" * 80)
    logger.info("Destroy target classifier comparison started.")
    logger.info("Config path: %s", config_path)
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
    screened_set = TargetScreener(load_target_screening_config(config)).screen(
        state.targets
    )

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

    test_size = float(
        get_config_value(
            config,
            "destroy_target_rf_training.test_size",
            default=0.25,
        )
    )
    trainer = SupervisedBaselineTrainer(
        test_size=test_size,
        random_seed=seed,
        logger=logger,
    )

    results: list[SupervisedBaselineResult] = []
    decision_tree_result = trainer.train(
        model_type="decision_tree",
        x=dataset.x,
        y=dataset.y,
        feature_names=feature_names,
        label_source=label_source,
        model_params={
            "max_depth": _optional_int(
                get_config_value(
                    config,
                    "destroy_target_classifier_comparison.decision_tree.max_depth",
                    default=14,
                )
            ),
            "random_state": seed,
        },
    )
    results.append(decision_tree_result)

    knn_result = trainer.train(
        model_type="knn",
        x=dataset.x,
        y=dataset.y,
        feature_names=feature_names,
        label_source=label_source,
        model_params={
            "n_neighbors": int(
                get_config_value(
                    config,
                    "destroy_target_classifier_comparison.knn.n_neighbors",
                    default=5,
                )
            ),
            "weights": str(
                get_config_value(
                    config,
                    "destroy_target_classifier_comparison.knn.weights",
                    default="distance",
                )
            ),
            "metric": str(
                get_config_value(
                    config,
                    "destroy_target_classifier_comparison.knn.metric",
                    default="minkowski",
                )
            ),
        },
    )
    results.append(knn_result)

    rf_result = _build_random_forest_result(
        config=config,
        project_root=project_root,
        trainer=trainer,
        x=dataset.x,
        y=dataset.y,
        feature_names=feature_names,
        label_source=label_source,
        seed=seed,
        logger=logger,
    )
    if rf_result is not None:
        results.append(rf_result)

    output_paths = _build_output_paths(config, project_root)
    save_comparison_csv(results, output_paths["comparison_csv"])
    save_model_report(decision_tree_result, output_paths["decision_tree_report"])
    save_model_report(knn_result, output_paths["knn_report"])

    plot_baseline_confusion_matrix(
        decision_tree_result.confusion_matrix,
        output_paths["decision_tree_confusion_matrix_figure"],
    )
    plot_baseline_confusion_matrix(
        knn_result.confusion_matrix,
        output_paths["knn_confusion_matrix_figure"],
    )
    plot_classifier_comparison_metrics(
        results,
        output_paths["comparison_metrics_figure"],
    )

    for result in results:
        logger.info(
            "%s metrics: accuracy=%.4f, macro_f1=%.4f",
            result.model_name,
            result.accuracy,
            result.f1_macro,
        )
    logger.info("Classifier comparison CSV saved to: %s", output_paths["comparison_csv"])
    logger.info("Decision Tree report saved to: %s", output_paths["decision_tree_report"])
    logger.info("KNN report saved to: %s", output_paths["knn_report"])
    logger.info(
        "Decision Tree confusion matrix figure saved to: %s",
        output_paths["decision_tree_confusion_matrix_figure"],
    )
    logger.info(
        "KNN confusion matrix figure saved to: %s",
        output_paths["knn_confusion_matrix_figure"],
    )
    logger.info(
        "Classifier comparison metrics figure saved to: %s",
        output_paths["comparison_metrics_figure"],
    )
    logger.info("Destroy target classifier comparison finished successfully.")


def _build_random_forest_result(
    *,
    config: dict,
    project_root: Path,
    trainer: SupervisedBaselineTrainer,
    x,
    y,
    feature_names: list[str],
    label_source: str,
    seed: int,
    logger,
) -> SupervisedBaselineResult | None:
    """加载已有 RF 模型评估；不存在时临时训练 RF baseline。"""
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
    if model_path.exists():
        model = load_random_forest_model(model_path)
        logger.info("Loaded RF model for comparison: %s", model_path)
        return trainer.evaluate_existing_model(
            model_name="RF",
            model=model,
            x=x,
            y=y,
            feature_names=feature_names,
            label_source=label_source,
        )

    logger.warning(
        "RF model not found at %s. Training an in-memory RF baseline for comparison.",
        model_path,
    )
    max_depth_raw = get_config_value(
        config,
        "destroy_target_rf_training.max_depth",
        default=None,
    )
    return trainer.train(
        model_type="random_forest",
        x=x,
        y=y,
        feature_names=feature_names,
        label_source=label_source,
        model_params={
            "n_estimators": int(
                get_config_value(
                    config,
                    "destroy_target_rf_training.n_estimators",
                    default=200,
                )
            ),
            "max_depth": _optional_int(max_depth_raw),
            "random_state": seed,
        },
    )


def _build_output_paths(config: dict, project_root: Path) -> dict[str, Path]:
    """读取并解析对比实验输出路径。"""
    prefix = "destroy_target_classifier_comparison.output"
    defaults = {
        "comparison_csv": "outputs/evaluation/destroy_target_classifier_comparison.csv",
        "decision_tree_report": "outputs/evaluation/baselines/decision_tree_report.txt",
        "knn_report": "outputs/evaluation/baselines/knn_report.txt",
        "decision_tree_confusion_matrix_figure": (
            "outputs/evaluation/figures/decision_tree_confusion_matrix.png"
        ),
        "knn_confusion_matrix_figure": (
            "outputs/evaluation/figures/knn_confusion_matrix.png"
        ),
        "comparison_metrics_figure": (
            "outputs/evaluation/figures/classifier_comparison_metrics.png"
        ),
    }
    return {
        key: resolve_path(
            str(get_config_value(config, f"{prefix}.{key}", default=default)),
            project_root=project_root,
        )
        for key, default in defaults.items()
    }


def _resolve_config_path(config_path: str | Path, project_root: Path) -> Path:
    """解析配置文件路径。"""
    path = Path(config_path)
    if path.is_absolute():
        return path
    return project_root / path


def _optional_int(value) -> int | None:
    """把 YAML 中的 null / 'null' / 数字转换为可选整数。"""
    if value in {None, "null"}:
        return None
    return int(value)


if __name__ == "__main__":
    main()

