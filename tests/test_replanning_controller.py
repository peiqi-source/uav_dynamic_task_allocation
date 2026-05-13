"""test重规划controller测试模块，用于验证对应业务模块的关键行为。"""
import subprocess
import sys


def test_replanning_controller_check_script_runs():
    """处理test重规划controllercheckscriptruns相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    result = subprocess.run(
        [sys.executable, "scripts/check_replanning_controller.py"],
        check=False,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
