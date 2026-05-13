"""Random Forest 摧毁目标集训练模块导入测试。"""


def test_destroy_target_rf_training_imports():
    """验证 RF 训练重构后的核心公开接口可以正常导入。"""
    from uav_dynamic_task_allocation.algorithms.supervised.random_forest.checkpoint import (
        load_random_forest_model,
        save_random_forest_model,
    )
    from uav_dynamic_task_allocation.algorithms.supervised.random_forest.trainer import (
        RandomForestDestroyTargetTrainer,
    )
    from uav_dynamic_task_allocation.training.destroy_target_rf_training import (
        run_destroy_target_rf_training,
    )

    assert RandomForestDestroyTargetTrainer is not None
    assert run_destroy_target_rf_training is not None
    assert save_random_forest_model is not None
    assert load_random_forest_model is not None

