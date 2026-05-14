"""KNN 摧毁目标集二分类 baseline。"""
from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier


class KNNDestroyTargetClassifier:
    """创建用于摧毁目标集决策的 KNeighborsClassifier。"""

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        weights: str = "distance",
        metric: str = "minkowski",
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def create_model(self) -> KNeighborsClassifier:
        """返回尚未训练的 KNeighborsClassifier。"""
        return KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
        )

