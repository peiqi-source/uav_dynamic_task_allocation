import subprocess
import sys


def test_replanning_controller_check_script_runs():
    result = subprocess.run(
        [sys.executable, "scripts/check_replanning_controller.py"],
        check=False,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
