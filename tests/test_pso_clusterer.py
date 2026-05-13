import numpy as np

from uav_dynamic_task_allocation.allocation.pso_clusterer import (
    PSOClusterer,
    PSOClustererConfig,
)


def test_pso_clusterer_returns_expected_shapes():
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
