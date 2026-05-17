"""Thin entry point for real-data PPO target regrouping training."""
from __future__ import annotations

import argparse

from uav_dynamic_task_allocation.training.ppo_clusterer_training import (
    run_ppo_clusterer_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO target regrouping policy.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ppo_clusterer_training(args.config)


if __name__ == "__main__":
    main()
