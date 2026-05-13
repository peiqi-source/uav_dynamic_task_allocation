import pandas as pd

from uav_dynamic_task_allocation.core.entities import build_battlefield_state


def test_build_battlefield_state_from_dataframes():
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
