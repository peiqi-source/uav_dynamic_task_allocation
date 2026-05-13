"""PPO 算法模块中的指标集合实现。"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


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
