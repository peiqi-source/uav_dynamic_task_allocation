"""监督学习 baseline 统一训练与评估。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any
import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from uav_dynamic_task_allocation.algorithms.supervised.baselines.decision_tree import (
    DecisionTreeDestroyTargetClassifier,
)
from uav_dynamic_task_allocation.algorithms.supervised.baselines.knn import (
    KNNDestroyTargetClassifier,
)


@dataclass(frozen=True)
class SupervisedBaselineResult:
    """单个监督学习分类器的训练/评估结果。"""

    model_name: str
    model: Any
    classification_report: str
    confusion_matrix: np.ndarray
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    train_size: int
    test_size: int
    use_holdout_split: bool
    label_source: str
    feature_names: list[str]


class SupervisedBaselineTrainer:
    """Decision Tree、KNN 和可选 Random Forest 的统一训练评估器。"""

    def __init__(
        self,
        *,
        test_size: float = 0.25,
        random_seed: int = 42,
        logger: logging.Logger | None = None,
    ) -> None:
        self.test_size = test_size
        self.random_seed = random_seed
        self.logger = logger or logging.getLogger(__name__)

    def train(
        self,
        *,
        model_type: str,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        label_source: str,
        model_params: dict[str, Any] | None = None,
    ) -> SupervisedBaselineResult:
        """创建、训练并评估指定类型的 baseline 模型。"""
        model = self._create_model(model_type, model_params or {})
        return self.train_existing_model(
            model_name=self._display_name(model_type),
            model=model,
            x=x,
            y=y,
            feature_names=feature_names,
            label_source=label_source,
            fit_model=True,
        )

    def evaluate_existing_model(
        self,
        *,
        model_name: str,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        label_source: str,
    ) -> SupervisedBaselineResult:
        """评估已存在的模型，不重新训练。"""
        return self.train_existing_model(
            model_name=model_name,
            model=model,
            x=x,
            y=y,
            feature_names=feature_names,
            label_source=label_source,
            fit_model=False,
        )

    def train_existing_model(
        self,
        *,
        model_name: str,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        label_source: str,
        fit_model: bool,
    ) -> SupervisedBaselineResult:
        """对给定模型执行统一 holdout 或全量训练评估流程。"""
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}")
        if len(x) != len(y):
            raise ValueError(f"x/y size mismatch: {len(x)} != {len(y)}")
        if x.shape[1] != len(feature_names):
            raise ValueError(
                f"feature_names length mismatch: {x.shape[1]} != {len(feature_names)}"
            )

        use_holdout = self.should_use_holdout_split(y, self.test_size)
        if use_holdout:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=self.test_size,
                random_state=self.random_seed,
                stratify=y,
            )
            if fit_model:
                model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            train_size = len(y_train)
            test_size = len(y_test)
            y_true = y_test
        else:
            if fit_model:
                model.fit(x, y)
            y_pred = model.predict(x)
            train_size = len(y)
            test_size = 0
            y_true = y
            self.logger.warning(
                "%s uses full-dataset training/evaluation because holdout split "
                "is unavailable for the current label distribution.",
                model_name,
            )

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Non-destroy", "Destroy"],
            digits=4,
            zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics = self._build_metrics(y_true, y_pred)

        return SupervisedBaselineResult(
            model_name=model_name,
            model=model,
            classification_report=report,
            confusion_matrix=matrix,
            accuracy=metrics["accuracy"],
            precision_macro=metrics["precision_macro"],
            recall_macro=metrics["recall_macro"],
            f1_macro=metrics["f1_macro"],
            precision_weighted=metrics["precision_weighted"],
            recall_weighted=metrics["recall_weighted"],
            f1_weighted=metrics["f1_weighted"],
            train_size=train_size,
            test_size=test_size,
            use_holdout_split=use_holdout,
            label_source=label_source,
            feature_names=feature_names,
        )

    @staticmethod
    def should_use_holdout_split(y: np.ndarray, test_size: float) -> bool:
        """判断当前标签分布是否可用 stratified holdout split。"""
        if len(y) < 8 or not 0.0 < test_size < 1.0:
            return False
        counts = Counter(int(value) for value in y.tolist())
        return len(counts) >= 2 and min(counts.values()) >= 2

    @staticmethod
    def _build_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """计算 accuracy、macro 和 weighted precision/recall/F1。"""
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
        }

    def _create_model(self, model_type: str, model_params: dict[str, Any]) -> Any:
        """按模型类型创建 sklearn 分类器。"""
        if model_type == "decision_tree":
            return DecisionTreeDestroyTargetClassifier(
                max_depth=model_params.get("max_depth", 14),
                random_state=model_params.get("random_state", self.random_seed),
                class_weight=model_params.get("class_weight", "balanced"),
            ).create_model()
        if model_type == "knn":
            return KNNDestroyTargetClassifier(
                n_neighbors=model_params.get("n_neighbors", 5),
                weights=model_params.get("weights", "distance"),
                metric=model_params.get("metric", "minkowski"),
            ).create_model()
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=model_params.get("n_estimators", 200),
                max_depth=model_params.get("max_depth"),
                random_state=model_params.get("random_state", self.random_seed),
                class_weight=model_params.get("class_weight", "balanced"),
            )
        raise ValueError(f"Unsupported baseline model_type: {model_type}")

    @staticmethod
    def _display_name(model_type: str) -> str:
        """把内部模型类型转换为报告中的名称。"""
        names = {
            "decision_tree": "Decision Tree",
            "knn": "KNN",
            "random_forest": "RF",
        }
        return names.get(model_type, model_type)

