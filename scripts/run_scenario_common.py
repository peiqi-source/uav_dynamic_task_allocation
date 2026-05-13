from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from uav_dynamic_task_allocation.utils.config import get_project_root


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def run_scenario(scenario_name: str) -> int:
    project_root = get_project_root()
    default_config_path = project_root / "configs" / "default.yaml"
    scenario_config_path = project_root / "configs" / "scenarios" / f"{scenario_name}.yaml"

    with default_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with scenario_config_path.open("r", encoding="utf-8") as file:
        scenario_config = yaml.safe_load(file)

    config = _deep_update(config, scenario_config)
    config.setdefault("full_simulation", {})
    config["full_simulation"]["output_root"] = "outputs/experiments"
    config["full_simulation"]["run_name"] = scenario_name
    config["full_simulation"]["create_timestamped_run_dir"] = False

    generated_dir = project_root / "outputs" / "experiments" / "_scenario_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_config_path = generated_dir / f"{scenario_name}.yaml"
    with generated_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    command = [
        sys.executable,
        "scripts/run_full_simulation.py",
        "--config",
        str(generated_config_path),
        "--run-name",
        scenario_name,
    ]
    return subprocess.call(command, cwd=project_root)
