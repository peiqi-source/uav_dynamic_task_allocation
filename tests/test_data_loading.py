"""test数据loading测试模块，用于验证对应业务模块的关键行为。"""
from pathlib import Path

from uav_dynamic_task_allocation.data.loaders import load_target_data, load_uav_data


def test_load_uav_and_target_csv(tmp_path: Path):
    """处理testload无人机and目标CSV 数据相关业务逻辑。

    参数：
        tmp_path: tmp路径，类型为 Path。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    uav_csv = tmp_path / "uav.csv"
    target_csv = tmp_path / "target.csv"
    uav_csv.write_text(
        "uav_id,x,y,uav_type,work_range,attack_power\n1,0,0,3,100,5\n",
        encoding="utf-8",
    )
    target_csv.write_text(
        "target_id,x,y,target_type,defense,significance\n1,10,0,1,2,3\n",
        encoding="utf-8",
    )

    assert load_uav_data(uav_csv).iloc[0]["uav_id"] == 1
    assert load_target_data(target_csv).iloc[0]["target_id"] == 1
