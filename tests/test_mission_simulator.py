import subprocess
import sys


def test_mission_simulator_check_script_runs():
    result = subprocess.run(
        [sys.executable, "scripts/check_mission_simulator.py"],
        check=False,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
