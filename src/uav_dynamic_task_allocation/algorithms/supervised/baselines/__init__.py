"""摧毁目标集监督学习 baseline 包。"""
from __future__ import annotations

from uav_dynamic_task_allocation.algorithms.supervised.baselines.decision_tree import (
    DecisionTreeDestroyTargetClassifier,
)
from uav_dynamic_task_allocation.algorithms.supervised.baselines.knn import (
    KNNDestroyTargetClassifier,
)
from uav_dynamic_task_allocation.algorithms.supervised.baselines.trainer import (
    SupervisedBaselineResult,
    SupervisedBaselineTrainer,
)

__all__ = [
    "DecisionTreeDestroyTargetClassifier",
    "KNNDestroyTargetClassifier",
    "SupervisedBaselineResult",
    "SupervisedBaselineTrainer",
]

