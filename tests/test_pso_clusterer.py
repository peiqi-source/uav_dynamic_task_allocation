"""testPSO 算法clusterer测试模块，用于验证对应业务模块的关键行为。"""
import numpy as np

from uav_dynamic_task_allocation.allocation.pso_clusterer import (
    PSOClusterer,
    PSOClustererConfig,
)


def test_pso_clusterer_returns_expected_shapes():
    """处理testPSO 算法clustererreturnsexpectedshapes相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    features = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.9, 1.0], [1.0, 0.9]],
        dtype=float,
    )
    clusterer = PSOClusterer(
        PSOClustererConfig(num_particles=5, max_iter=5, random_seed=7)
    )

    result = clusterer.cluster(features=features, num_clusters=2)

    assert result.labels.shape == (4,)
    assert result.centers.shape == (2, 2)
    assert result.best_score >= 0.0
