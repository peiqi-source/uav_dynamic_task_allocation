"""Random Forest 摧毁目标集决策模型训练器。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from uav_dynamic_task_allocation.algorithms.supervised.random_forest.metrics import (
    save_feature_importance,
)


@dataclass(frozen=True)
class RandomForestTrainingResult:
    """
    Random Forest 训练结果。

    属性：
        model: 训练完成的 RandomForestClassifier。
        classification_report: sklearn classification_report 文本。
        confusion_matrix: sklearn confusion_matrix 矩阵。
        feature_names: 模型输入特征名，顺序与训练矩阵列一致。
        label_counts: 各标签样本数量。
        use_holdout_split: 是否使用 holdout train/test split。
        train_size: 训练样本数量。
        test_size: 测试样本数量；全量训练时为 0。
        metadata: 训练过程中的补充信息。
    """

    model: RandomForestClassifier
    classification_report: str
    confusion_matrix: np.ndarray
    feature_names: list[str]
    label_counts: dict[int, int]
    use_holdout_split: bool
    train_size: int
    test_size: int
    metadata: dict[str, Any]


class RandomForestDestroyTargetTrainer:
    """
    摧毁目标集 Random Forest 训练器。

    该类只负责 sklearn 模型训练和评估，不负责读取项目配置、加载数据或生成标签。
    """

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = None,
        test_size: float = 0.25,
        random_seed: int = 42,
        class_weight: str | dict[int, float] | None = "balanced",
        logger: logging.Logger | None = None,
    ) -> None:
        """
        初始化训练器。

        参数：
            n_estimators: 随机森林树的数量。
            max_depth: 每棵树的最大深度；None 表示不限制。
            test_size: holdout 测试集比例。
            random_seed: 随机种子。
            class_weight: sklearn class_weight 参数。
            logger: 可选日志器。
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_seed = random_seed
        self.class_weight = class_weight
        self.logger = logger or logging.getLogger(__name__)

    def create_model(self) -> RandomForestClassifier:
        """
        创建 RandomForestClassifier。

        返回：
            尚未训练的 RandomForestClassifier。
        """
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_seed,
            class_weight=self.class_weight,
        )

    def train(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        feature_importance_path: str | Path | None = None,
    ) -> RandomForestTrainingResult:
        """
        训练 Random Forest 并生成评估结果。

        参数：
            x: 训练特征矩阵。
            y: 二分类标签数组。
            feature_names: 与 x 列顺序一致的特征名。
            feature_importance_path: 可选；提供时保存特征重要性 CSV。

        返回：
            RandomForestTrainingResult。
        """
        if x.ndim != 2:
            raise ValueError(f"x must be a 2D matrix, got shape={x.shape}")
        if len(x) != len(y):
            raise ValueError(f"x/y size mismatch: len(x)={len(x)}, len(y)={len(y)}")
        if x.shape[1] != len(feature_names):
            raise ValueError(
                "feature_names length must match x columns: "
                f"columns={x.shape[1]}, feature_names={len(feature_names)}"
            )
        if len(y) == 0:
            raise ValueError("Random Forest training requires at least one sample.")

        label_counts = dict(Counter(int(value) for value in y.tolist()))
        use_holdout = self.should_use_holdout_split(y=y, test_size=self.test_size)
        model = self.create_model()

        if use_holdout:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=self.test_size,
                random_state=self.random_seed,
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
            train_size = len(y_train)
            test_size = len(y_test)

            self.logger.info("Using holdout train/test split.")
            self.logger.info("Train size: %s, test size: %s", train_size, test_size)
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
            train_size = len(y)
            test_size = 0

            self.logger.warning(
                "Dataset is too small or class distribution is imbalanced. "
                "Training and reporting on the full dataset. "
                "This is acceptable for engineering validation, but not for final evaluation."
            )

        if feature_importance_path is not None:
            save_feature_importance(
                model=model,
                feature_names=feature_names,
                output_path=feature_importance_path,
            )

        return RandomForestTrainingResult(
            model=model,
            classification_report=report,
            confusion_matrix=matrix,
            feature_names=feature_names,
            label_counts=label_counts,
            use_holdout_split=use_holdout,
            train_size=train_size,
            test_size=test_size,
            metadata={
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_seed": self.random_seed,
                "class_weight": self.class_weight,
            },
        )

    @staticmethod
    def should_use_holdout_split(y: np.ndarray, test_size: float) -> bool:
        """
        判断当前数据是否适合划分 train/test。

        参数：
            y: 标签数组。
            test_size: 测试集比例。

        返回：
            True 表示可以使用 stratified holdout split；False 表示应全量训练。
        """
        if len(y) < 8:
            return False
        if not 0.0 < test_size < 1.0:
            return False

        counts = Counter(int(value) for value in y.tolist())
        if len(counts) < 2:
            return False
        if min(counts.values()) < 2:
            return False

        return True

