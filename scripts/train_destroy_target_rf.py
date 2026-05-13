"""Random Forest 摧毁目标集训练命令行入口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from uav_dynamic_task_allocation.training.destroy_target_rf_training import (  # noqa: E402
    run_destroy_target_rf_training,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train Random Forest model for destroy target selection."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    """调用 src 中的可复用训练流水线。"""
    args = parse_args()
    run_destroy_target_rf_training(args.config)


if __name__ == "__main__":
    main()
