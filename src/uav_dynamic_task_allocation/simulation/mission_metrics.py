"""仿真模块中的任务指标集合实现。"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    resolve_path,
)


class MissionMetricsError(Exception):
    """任务仿真指标统计与可视化过程中的自定义错误。"""


@dataclass(frozen=True)
class MissionMetricsConfig:
    """
    任务仿真指标分析配置。

    input_csv_path:
        MissionSimulator 输出的时间步日志。

    summary_csv_path:
        保存仿真总体指标。

    timeline_csv_path:
        保存每个时间步的指标展开结果。

    figure_dir:
        保存趋势图。
    """

    # input_csv_path: 输入CSV 数据路径。
    input_csv_path: str = "outputs/simulation/mission_simulation_log.csv"
    # summary_csv_path: summaryCSV 数据路径。
    summary_csv_path: str = "outputs/simulation/mission_simulation_summary.csv"
    # timeline_csv_path: timelineCSV 数据路径。
    timeline_csv_path: str = "outputs/simulation/mission_simulation_timeline_metrics.csv"
    # figure_dir: 图表dir。
    figure_dir: str = "outputs/simulation/figures"

    def validate(self) -> None:
        """校验当前对象或输入配置的合法性。

        参数：
            无显式业务参数。

        返回：
            无返回值；通过状态变更、文件输出或日志记录体现执行结果。
        """
        if not self.input_csv_path:
            raise MissionMetricsError("input_csv_path must not be empty.")

        if not self.summary_csv_path:
            raise MissionMetricsError("summary_csv_path must not be empty.")

        if not self.timeline_csv_path:
            raise MissionMetricsError("timeline_csv_path must not be empty.")

        if not self.figure_dir:
            raise MissionMetricsError("figure_dir must not be empty.")


@dataclass
class MissionMetricsResult:
    """
    仿真指标分析结果。

    summary:
        总体指标。

    timeline:
        时间步级指标表。
    """

    # summary: summary 数据。
    summary: dict[str, Any]
    # timeline: timeline 数据。
    timeline: pd.DataFrame


class MissionMetricsAnalyzer:
    """
    任务仿真结果分析器。

    输入：
        MissionSimulator 生成的 simulation_log.csv

    输出：
        1. 总体 summary
        2. 时间步 timeline metrics
        3. 若干趋势图

    该模块不重新运行仿真，只分析已有日志。
    """

    def __init__(self, config: MissionMetricsConfig) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            config: 配置对象，类型为 MissionMetricsConfig。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        config.validate()
        # config: 配置。
        self.config = config

    def analyze(
        self,
        project_root: str | Path | None = None,
    ) -> MissionMetricsResult:
        """
        分析仿真日志。
        """
        input_path = resolve_path(
            self.config.input_csv_path,
            project_root=project_root,
        )

        if not input_path.exists():
            raise MissionMetricsError(
                f"Simulation log CSV not found: {input_path}. "
                "Please run scripts/check_mission_simulator.py first."
            )

        raw_df = pd.read_csv(input_path, encoding="utf-8-sig")

        if raw_df.empty:
            raise MissionMetricsError(f"Simulation log CSV is empty: {input_path}")

        timeline = self._build_timeline_metrics(raw_df)
        summary = self._build_summary(timeline)

        return MissionMetricsResult(
            summary=summary,
            timeline=timeline,
        )

    def write_outputs(
        self,
        result: MissionMetricsResult,
        project_root: str | Path | None = None,
    ) -> tuple[Path, Path]:
        """
        保存 summary 和 timeline 指标。
        """
        summary_path = resolve_path(
            self.config.summary_csv_path,
            project_root=project_root,
        )
        timeline_path = resolve_path(
            self.config.timeline_csv_path,
            project_root=project_root,
        )

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        timeline_path.parent.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame([result.summary])
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        result.timeline.to_csv(
            timeline_path,
            index=False,
            encoding="utf-8-sig",
        )

        return summary_path, timeline_path

    def save_plots(
        self,
        result: MissionMetricsResult,
        project_root: str | Path | None = None,
    ) -> list[Path]:
        """
        保存仿真趋势图。

        当前每张图单独保存，便于后续报告和论文插图使用。
        """
        figure_dir = resolve_path(
            self.config.figure_dir,
            project_root=project_root,
        )
        figure_dir.mkdir(parents=True, exist_ok=True)

        timeline = result.timeline

        output_paths: list[Path] = []

        output_paths.append(
            self._save_line_plot(
                timeline=timeline,
                y_column="destroyed_count",
                output_path=figure_dir / "destroyed_targets_over_time.png",
                title="Destroyed Targets Over Time",
                ylabel="Destroyed Target Count",
            )
        )

        output_paths.append(
            self._save_line_plot(
                timeline=timeline,
                y_column="available_target_count",
                output_path=figure_dir / "available_targets_over_time.png",
                title="Available Targets Over Time",
                ylabel="Available Target Count",
            )
        )

        output_paths.append(
            self._save_line_plot(
                timeline=timeline,
                y_column="available_uav_count",
                output_path=figure_dir / "available_uavs_over_time.png",
                title="Available UAVs Over Time",
                ylabel="Available UAV Count",
            )
        )

        output_paths.append(
            self._save_line_plot(
                timeline=timeline,
                y_column="replanning_count",
                output_path=figure_dir / "replanning_count_over_time.png",
                title="Replanning Count Over Time",
                ylabel="Replanning Count",
            )
        )

        output_paths.append(
            self._save_line_plot(
                timeline=timeline,
                y_column="completed_cluster_ratio",
                output_path=figure_dir / "cluster_completion_ratio_over_time.png",
                title="Cluster Completion Ratio Over Time",
                ylabel="Completed Cluster Ratio",
            )
        )

        output_paths.append(
            self._save_line_plot(
                timeline=timeline,
                y_column="event_count",
                output_path=figure_dir / "event_count_over_time.png",
                title="Dynamic Event Count Over Time",
                ylabel="Event Count in Step",
            )
        )

        return output_paths

    def _build_timeline_metrics(
        self,
        raw_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        将原始 simulation log 展开成更适合分析的时间步指标表。

        原始 CSV 中很多字段是 list / dict 的字符串表示，
        这里会解析成数量和 JSON 字符串。
        """
        required_columns = [
            "time",
            "event_types",
            "replanning_count",
            "destroyed_target_ids",
            "disappeared_target_ids",
            "damaged_uav_ids",
            "available_target_count",
            "available_uav_count",
            "completed_cluster_count",
            "total_cluster_count",
            "metadata",
        ]

        missing_columns = [
            column for column in required_columns
            if column not in raw_df.columns
        ]

        if missing_columns:
            raise MissionMetricsError(
                f"Simulation log missing columns: {missing_columns}"
            )

        rows: list[dict[str, Any]] = []

        for _, row in raw_df.iterrows():
            event_types = self._parse_list(row["event_types"])
            destroyed_ids = self._parse_list(row["destroyed_target_ids"])
            disappeared_ids = self._parse_list(row["disappeared_target_ids"])
            damaged_uav_ids = self._parse_list(row["damaged_uav_ids"])
            metadata = self._parse_dict(row["metadata"])

            destroyed_this_step = metadata.get("destroyed_this_step", [])
            cluster_status = metadata.get("cluster_status", {})

            total_cluster_count = int(row["total_cluster_count"])
            completed_cluster_count = int(row["completed_cluster_count"])

            completed_cluster_ratio = (
                completed_cluster_count / total_cluster_count
                if total_cluster_count > 0
                else 0.0
            )

            rows.append(
                {
                    "time": float(row["time"]),
                    "event_count": len(event_types),
                    "event_types": json.dumps(event_types, ensure_ascii=False),
                    "replanning_count": int(row["replanning_count"]),
                    "destroyed_count": len(destroyed_ids),
                    "destroyed_target_ids": json.dumps(
                        destroyed_ids,
                        ensure_ascii=False,
                    ),
                    "destroyed_this_step_count": len(destroyed_this_step),
                    "destroyed_this_step": json.dumps(
                        destroyed_this_step,
                        ensure_ascii=False,
                    ),
                    "disappeared_count": len(disappeared_ids),
                    "disappeared_target_ids": json.dumps(
                        disappeared_ids,
                        ensure_ascii=False,
                    ),
                    "damaged_uav_count": len(damaged_uav_ids),
                    "damaged_uav_ids": json.dumps(
                        damaged_uav_ids,
                        ensure_ascii=False,
                    ),
                    "available_target_count": int(row["available_target_count"]),
                    "available_uav_count": int(row["available_uav_count"]),
                    "completed_cluster_count": completed_cluster_count,
                    "total_cluster_count": total_cluster_count,
                    "completed_cluster_ratio": completed_cluster_ratio,
                    "cluster_status": json.dumps(
                        cluster_status,
                        ensure_ascii=False,
                    ),
                }
            )

        timeline = pd.DataFrame(rows)
        timeline = timeline.sort_values("time").reset_index(drop=True)

        return timeline

    def _build_summary(
        self,
        timeline: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        根据 timeline 构造总体仿真指标。
        """
        final_row = timeline.iloc[-1]

        event_type_counts: dict[str, int] = {}

        for value in timeline["event_types"]:
            event_types = self._parse_list(value)

            for event_type in event_types:
                event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

        final_destroyed_ids = self._parse_list(final_row["destroyed_target_ids"])
        final_disappeared_ids = self._parse_list(final_row["disappeared_target_ids"])
        final_damaged_uav_ids = self._parse_list(final_row["damaged_uav_ids"])

        final_total_cluster_count = int(final_row["total_cluster_count"])
        final_completed_cluster_count = int(final_row["completed_cluster_count"])

        final_cluster_completion_ratio = (
            final_completed_cluster_count / final_total_cluster_count
            if final_total_cluster_count > 0
            else 0.0
        )

        total_destroyed_this_step = int(
            timeline["destroyed_this_step_count"].sum()
        )

        summary = {
            "total_time": float(final_row["time"]),
            "num_steps": int(len(timeline)),
            "num_replanning": int(final_row["replanning_count"]),
            "total_event_count": int(timeline["event_count"].sum()),
            "event_type_counts": json.dumps(
                event_type_counts,
                ensure_ascii=False,
            ),
            "final_destroyed_count": len(final_destroyed_ids),
            "final_destroyed_target_ids": json.dumps(
                final_destroyed_ids,
                ensure_ascii=False,
            ),
            "final_disappeared_count": len(final_disappeared_ids),
            "final_disappeared_target_ids": json.dumps(
                final_disappeared_ids,
                ensure_ascii=False,
            ),
            "final_damaged_uav_count": len(final_damaged_uav_ids),
            "final_damaged_uav_ids": json.dumps(
                final_damaged_uav_ids,
                ensure_ascii=False,
            ),
            "final_available_target_count": int(
                final_row["available_target_count"]
            ),
            "final_available_uav_count": int(
                final_row["available_uav_count"]
            ),
            "final_completed_cluster_count": final_completed_cluster_count,
            "final_total_cluster_count": final_total_cluster_count,
            "final_cluster_completion_ratio": final_cluster_completion_ratio,
            "mission_completed_by_cluster": (
                final_total_cluster_count > 0
                and final_completed_cluster_count == final_total_cluster_count
            ),
            "total_destroyed_this_step_sum": total_destroyed_this_step,
            "average_destroyed_per_step": (
                total_destroyed_this_step / len(timeline)
                if len(timeline) > 0
                else 0.0
            ),
        }

        return summary

    def _save_line_plot(
        self,
        timeline: pd.DataFrame,
        y_column: str,
        output_path: Path,
        title: str,
        ylabel: str,
    ) -> Path:
        """
        保存单条时间序列图。
        """
        if y_column not in timeline.columns:
            raise MissionMetricsError(f"Missing timeline column: {y_column}")

        plt.figure(figsize=(10, 6))
        plt.plot(timeline["time"], timeline[y_column], linewidth=2.0)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()

        return output_path

    @staticmethod
    def _parse_list(value: Any) -> list[Any]:
        """
        解析 CSV 中保存的 list 字段。
        """
        if isinstance(value, list):
            return value

        if pd.isna(value):
            return []

        value_str = str(value).strip()

        if not value_str:
            return []

        try:
            parsed = ast.literal_eval(value_str)
        except Exception:
            try:
                parsed = json.loads(value_str)
            except Exception:
                return [value_str]

        if isinstance(parsed, list):
            return parsed

        return [parsed]

    @staticmethod
    def _parse_dict(value: Any) -> dict[str, Any]:
        """
        解析 CSV 中保存的 dict 字段。
        """
        if isinstance(value, dict):
            return value

        if pd.isna(value):
            return {}

        value_str = str(value).strip()

        if not value_str:
            return {}

        try:
            parsed = ast.literal_eval(value_str)
        except Exception:
            try:
                parsed = json.loads(value_str)
            except Exception:
                return {}

        if isinstance(parsed, dict):
            return parsed

        return {}


def load_mission_metrics_config(
    config: dict[str, Any],
) -> MissionMetricsConfig:
    """从项目总配置中读取 MissionMetricsConfig。"""
    metrics_config = MissionMetricsConfig(
        input_csv_path=str(
            get_config_value(
                config,
                "mission_metrics.input_csv_path",
                default="outputs/simulation/mission_simulation_log.csv",
            )
        ),
        summary_csv_path=str(
            get_config_value(
                config,
                "mission_metrics.output.summary_csv_path",
                default="outputs/simulation/mission_simulation_summary.csv",
            )
        ),
        timeline_csv_path=str(
            get_config_value(
                config,
                "mission_metrics.output.timeline_csv_path",
                default="outputs/simulation/mission_simulation_timeline_metrics.csv",
            )
        ),
        figure_dir=str(
            get_config_value(
                config,
                "mission_metrics.output.figure_dir",
                default="outputs/simulation/figures",
            )
        ),
    )

    metrics_config.validate()
    return metrics_config


def build_mission_metrics_analyzer(
    config: dict[str, Any],
) -> MissionMetricsAnalyzer:
    """构造 MissionMetricsAnalyzer。"""
    return MissionMetricsAnalyzer(
        config=load_mission_metrics_config(config)
    )