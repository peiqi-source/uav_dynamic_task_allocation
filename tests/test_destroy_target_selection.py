"""testdestroy目标selection测试模块，用于验证对应业务模块的关键行为。"""
from uav_dynamic_task_allocation.core.contracts import ScreenedTargetSet, TargetScore
from uav_dynamic_task_allocation.core.entities import Position, Target
from uav_dynamic_task_allocation.preprocessing.destroy_target_selection import (
    DestroyTargetSelectionConfig,
    DestroyTargetSelector,
)


def test_top_k_destroy_target_selection():
    """处理testtopkdestroy目标selection相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    targets = [
        Target(i, Position(float(i), 0.0), 1, defense=1.0, significance=float(i))
        for i in range(1, 5)
    ]
    scores = {
        target.target_id: TargetScore(
            target_id=target.target_id,
            score=float(target.target_id),
            components={"normalized_distance": 0.1},
        )
        for target in targets
    }
    screened = ScreenedTargetSet(
        all_targets=targets,
        candidate_targets=targets,
        selected_targets=targets,
        target_scores=scores,
    )

    result = DestroyTargetSelector(
        DestroyTargetSelectionConfig(method="top_k", rule_top_k=2)
    ).select(screened)

    assert [target.target_id for target in result.destroy_targets] == [4, 3]
