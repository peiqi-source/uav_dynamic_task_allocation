"""Decision Tree 摧毁目标集二分类 baseline。"""
from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier


class DecisionTreeDestroyTargetClassifier:
    """创建用于摧毁目标集决策的 DecisionTreeClassifier。"""

    def __init__(
        self,
        *,
        max_depth: int | None = 14,
        random_state: int = 42,
        class_weight: str | dict[int, float] | None = "balanced",
    ) -> None:
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weight = class_weight

    def create_model(self) -> DecisionTreeClassifier:
        """返回尚未训练的 DecisionTreeClassifier。"""
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )

