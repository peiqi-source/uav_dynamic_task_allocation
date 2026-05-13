"""testentities测试模块，用于验证对应业务模块的关键行为。"""
import pandas as pd

from uav_dynamic_task_allocation.core.entities import build_battlefield_state


def test_build_battlefield_state_from_dataframes():
    """处理testbuild战场状态fromdataframes相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    uav_df = pd.DataFrame(
        [
            {"uav_id": 1, "x": 0, "y": 0, "uav_type": 3, "work_range": 100, "attack_power": 5},
            {"uav_id": 2, "x": 1, "y": 0, "uav_type": 1, "work_range": 100, "attack_power": 0},
        ]
    )
    target_df = pd.DataFrame(
        [{"target_id": 1, "x": 10, "y": 0, "target_type": 1, "defense": 2, "significance": 3}]
    )

    state = build_battlefield_state(uav_df=uav_df, target_df=target_df)

    assert len(state.uavs) == 2
    assert len(state.targets) == 1
    assert state.attack_uavs[0].uav_id == 1
