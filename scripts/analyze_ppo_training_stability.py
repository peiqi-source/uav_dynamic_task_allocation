"""Analyze PPO training stability from metrics CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    get_project_root,
    load_and_validate_config,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PPO training stability.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_and_validate_config(config_path)

    metrics_csv = resolve_path(
        get_config_value(
            config,
            "ppo_clusterer_training.output.training_metrics_csv",
            "outputs/metrics/ppo_clusterer_training_metrics.csv",
        ),
        project_root=project_root,
    )
    report_path = resolve_path(
        "outputs/evaluation/ppo_training_stability_report.txt",
        project_root=project_root,
    )
    by_event_path = resolve_path(
        "outputs/evaluation/ppo_training_stability_by_event.csv",
        project_root=project_root,
    )

    rows = _read_rows(metrics_csv)
    if not rows:
        raise RuntimeError(f"No PPO metrics rows found: {metrics_csv}")

    reward_values = _float_values(rows, "total_reward")
    score_values = _float_values(rows, "final_score")

    first_rewards = reward_values[:100]
    last_rewards = reward_values[-100:]
    first_scores = score_values[:100]
    last_scores = score_values[-100:]

    reward_first_mean, reward_first_std = _mean_std(first_rewards)
    reward_last_mean, reward_last_std = _mean_std(last_rewards)
    score_first_mean, score_first_std = _mean_std(first_scores)
    score_last_mean, score_last_std = _mean_std(last_scores)

    by_event_rows = _build_by_event_rows(rows)
    _write_by_event_csv(by_event_rows, by_event_path)

    lines = [
        "PPO Training Stability Report",
        f"metrics_csv: {metrics_csv}",
        f"num_episodes: {len(rows)}",
        "",
        f"first_100_reward_mean: {reward_first_mean:.6f}",
        f"first_100_reward_std: {reward_first_std:.6f}",
        f"last_100_reward_mean: {reward_last_mean:.6f}",
        f"last_100_reward_std: {reward_last_std:.6f}",
        f"reward_improvement: {reward_last_mean - reward_first_mean:.6f}",
        "",
        f"first_100_score_mean: {score_first_mean:.6f}",
        f"first_100_score_std: {score_first_std:.6f}",
        f"last_100_score_mean: {score_last_mean:.6f}",
        f"last_100_score_std: {score_last_std:.6f}",
        f"score_improvement: {score_last_mean - score_first_mean:.6f}",
        "",
        "By event type:",
    ]
    for row in by_event_rows:
        lines.append(
            "event_type={event_type}, count={count}, "
            "reward_mean={reward_mean:.6f}, reward_std={reward_std:.6f}, "
            "score_mean={score_mean:.6f}, score_std={score_std:.6f}".format(**row)
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PPO stability report saved to: {report_path}")
    print(f"PPO stability by-event CSV saved to: {by_event_path}")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw_value = row.get(key, "")
        if raw_value in {"", None}:
            continue
        values.append(float(raw_value))
    return values


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(mean(values)), float(pstdev(values)) if len(values) > 1 else 0.0


def _build_by_event_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        event_type = row.get("event_type", "unknown") or "unknown"
        grouped.setdefault(event_type, []).append(row)

    output_rows: list[dict[str, Any]] = []
    for event_type in sorted(grouped):
        event_rows = grouped[event_type]
        reward_mean, reward_std = _mean_std(_float_values(event_rows, "total_reward"))
        score_mean, score_std = _mean_std(_float_values(event_rows, "final_score"))
        output_rows.append(
            {
                "event_type": event_type,
                "count": len(event_rows),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "score_mean": score_mean,
                "score_std": score_std,
            }
        )
    return output_rows


def _write_by_event_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "event_type",
            "count",
            "reward_mean",
            "reward_std",
            "score_mean",
            "score_std",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
