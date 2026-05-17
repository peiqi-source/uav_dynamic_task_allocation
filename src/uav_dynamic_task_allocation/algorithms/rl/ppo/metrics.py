"""PPO 算法模块中的指标集合实现。"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def add_moving_average_metrics(
    rows: list[dict[str, Any]],
    reward_key: str = "total_reward",
    score_key: str = "final_score",
) -> list[dict[str, Any]]:
    """Add moving-average reward and score columns without removing existing fields."""
    enriched = [dict(row) for row in rows]
    _add_moving_average(enriched, reward_key, "moving_avg_reward_20", 20)
    _add_moving_average(enriched, score_key, "moving_avg_score_20", 20)
    _add_moving_average(enriched, reward_key, "moving_avg_reward_50", 50)
    _add_moving_average(enriched, score_key, "moving_avg_score_50", 50)
    return enriched


def write_ppo_metrics(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write PPO training/evaluation metrics to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["metric"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _add_moving_average(
    rows: list[dict[str, Any]],
    source_key: str,
    output_key: str,
    window: int,
) -> None:
    values: list[float] = []
    for row in rows:
        raw_value = row.get(source_key, 0.0)
        values.append(float(raw_value) if raw_value not in {"", None} else 0.0)

    for index, row in enumerate(rows):
        start = max(0, index - window + 1)
        window_values = values[start:index + 1]
        row[output_key] = (
            sum(window_values) / len(window_values)
            if window_values
            else 0.0
        )
