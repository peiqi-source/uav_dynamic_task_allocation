"""runallexperiments脚本，封装可直接运行的实验、检查或可视化流程。"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from uav_dynamic_task_allocation.utils.config import get_project_root


SCENARIO_SCRIPTS = {
    "static": "scripts/run_static_scenario.py",
    "target_removed": "scripts/run_target_removed_scenario.py",
    "target_added": "scripts/run_target_added_scenario.py",
    "uav_lost": "scripts/run_uav_lost_scenario.py",
    "comprehensive_dynamic": "scripts/run_comprehensive_dynamic_scenario.py",
}


def main() -> int:
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        int，表示该函数计算或构建得到的结果。
    """
    project_root = get_project_root()
    rows = []
    for scenario_name, script_path in SCENARIO_SCRIPTS.items():
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            check=False,
        )
        summary_path = (
            project_root
            / "outputs"
            / "experiments"
            / scenario_name
            / "final_summary.json"
        )
        summary = {}
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as file:
                summary = json.load(file)
        rows.append(
            {
                "scenario": scenario_name,
                "returncode": result.returncode,
                "summary_path": str(summary_path),
                "num_replanning_results": summary.get("num_replanning_results", ""),
                "metrics_summary": summary.get("metrics_summary", {}),
            }
        )

    output_path = project_root / "outputs" / "experiments" / "all_experiments_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scenario",
                "returncode",
                "summary_path",
                "num_replanning_results",
                "metrics_summary",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return 0 if all(row["returncode"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
