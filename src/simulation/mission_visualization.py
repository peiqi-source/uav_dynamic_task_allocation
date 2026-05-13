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


class MissionVisualizationError(Exception):
    """任务仿真可视化过程中的自定义错误。"""


@dataclass(frozen=True)
class MissionVisualizationConfig:
    """
    任务仿真可视化配置。

    uav_trajectory_csv_path:
        MissionSimulator 输出的单架 UAV 轨迹日志。

    simulation_log_csv_path:
        MissionSimulator 输出的时间步日志。

    figure_dir:
        保存图片的目录。

    max_uavs_to_plot:
        空间轨迹图中最多绘制多少架 UAV，避免图太乱。

    only_assigned_uavs:
        是否只绘制参与任务分配的 UAV。
    """

    uav_trajectory_csv_path: str = "outputs/simulation/uav_trajectory_log.csv"
    simulation_log_csv_path: str = "outputs/simulation/mission_simulation_log.csv"
    figure_dir: str = "outputs/simulation/figures"

    max_uavs_to_plot: int = 20
    only_assigned_uavs: bool = True

    def validate(self) -> None:
        if not self.uav_trajectory_csv_path:
            raise MissionVisualizationError("uav_trajectory_csv_path must not be empty.")

        if not self.simulation_log_csv_path:
            raise MissionVisualizationError("simulation_log_csv_path must not be empty.")

        if not self.figure_dir:
            raise MissionVisualizationError("figure_dir must not be empty.")

        if self.max_uavs_to_plot <= 0:
            raise MissionVisualizationError("max_uavs_to_plot must be positive.")


class MissionVisualizer:
    """
    任务仿真可视化器。

    输入：
        uav_trajectory_log.csv
        mission_simulation_log.csv
        battlefield_state.targets

    输出：
        UAV 空间轨迹图
        UAV 状态时间线图
        动态事件 / 重规划时间线图
        目标最终状态空间图
    """

    def __init__(self, config: MissionVisualizationConfig) -> None:
        config.validate()
        self.config = config

    def save_all(
        self,
        targets: list[Any],
        project_root: str | Path | None = None,
    ) -> list[Path]:
        """
        保存所有仿真可视化图片。
        """
        trajectory_df, simulation_df = self._load_inputs(project_root=project_root)

        figure_dir = resolve_path(
            self.config.figure_dir,
            project_root=project_root,
        )
        figure_dir.mkdir(parents=True, exist_ok=True)

        target_df = self._build_target_dataframe(targets)
        selected_uav_ids = self._select_uav_ids(trajectory_df)

        output_paths = [
            self._save_uav_spatial_trajectory_plot(
                trajectory_df=trajectory_df,
                target_df=target_df,
                selected_uav_ids=selected_uav_ids,
                simulation_df=simulation_df,
                output_path=figure_dir / "uav_spatial_trajectories.png",
            ),
            self._save_uav_status_timeline_plot(
                trajectory_df=trajectory_df,
                selected_uav_ids=selected_uav_ids,
                output_path=figure_dir / "uav_status_timeline.png",
            ),
            self._save_event_replanning_timeline_plot(
                simulation_df=simulation_df,
                output_path=figure_dir / "event_replanning_timeline.png",
            ),
            self._save_target_final_status_plot(
                target_df=target_df,
                simulation_df=simulation_df,
                output_path=figure_dir / "target_final_status_map.png",
            ),
        ]

        return output_paths

    def _load_inputs(
        self,
        project_root: str | Path | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """读取轨迹日志和仿真日志。"""
        trajectory_path = resolve_path(
            self.config.uav_trajectory_csv_path,
            project_root=project_root,
        )
        simulation_path = resolve_path(
            self.config.simulation_log_csv_path,
            project_root=project_root,
        )

        if not trajectory_path.exists():
            raise MissionVisualizationError(
                f"UAV trajectory CSV not found: {trajectory_path}. "
                "Please run scripts/check_mission_simulator.py first."
            )

        if not simulation_path.exists():
            raise MissionVisualizationError(
                f"Simulation log CSV not found: {simulation_path}. "
                "Please run scripts/check_mission_simulator.py first."
            )

        trajectory_df = pd.read_csv(trajectory_path, encoding="utf-8-sig")
        simulation_df = pd.read_csv(simulation_path, encoding="utf-8-sig")

        if trajectory_df.empty:
            raise MissionVisualizationError(
                f"UAV trajectory CSV is empty: {trajectory_path}"
            )

        if simulation_df.empty:
            raise MissionVisualizationError(
                f"Simulation log CSV is empty: {simulation_path}"
            )

        return trajectory_df, simulation_df

    def _build_target_dataframe(
        self,
        targets: list[Any],
    ) -> pd.DataFrame:
        """把 Target 实体转换成可绘图 DataFrame。"""
        rows = []

        for target in targets:
            rows.append(
                {
                    "target_id": int(target.target_id),
                    "x": float(target.position.x),
                    "y": float(target.position.y),
                }
            )

        return pd.DataFrame(rows)

    def _select_uav_ids(
        self,
        trajectory_df: pd.DataFrame,
    ) -> list[int]:
        """
        选择需要绘图的 UAV。

        优先选择：
        - assigned_cluster_id 不为空的 UAV；
        - 或状态不是 available 的 UAV；
        - 或位置发生明显变化的 UAV。
        """
        df = trajectory_df.copy()

        selected_ids: set[int] = set()

        if self.config.only_assigned_uavs and "assigned_cluster_id" in df.columns:
            assigned_df = df[df["assigned_cluster_id"].notna()]
            selected_ids.update(int(value) for value in assigned_df["uav_id"].unique())

        active_df = df[df["status"].astype(str) != "available"]
        selected_ids.update(int(value) for value in active_df["uav_id"].unique())

        for uav_id, group in df.groupby("uav_id"):
            x_range = float(group["x"].max() - group["x"].min())
            y_range = float(group["y"].max() - group["y"].min())

            if abs(x_range) > 1e-6 or abs(y_range) > 1e-6:
                selected_ids.add(int(uav_id))

        if not selected_ids:
            selected_ids.update(int(value) for value in df["uav_id"].unique())

        selected = sorted(selected_ids)

        return selected[: self.config.max_uavs_to_plot]

    def _save_uav_spatial_trajectory_plot(
        self,
        trajectory_df: pd.DataFrame,
        target_df: pd.DataFrame,
        selected_uav_ids: list[int],
        simulation_df: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """
        绘制 UAV 空间轨迹图。

        图中包括：
        - 所有 target 的空间位置；
        - 被选中 UAV 的运动轨迹；
        - 每架 UAV 的起点和终点；
        - damaged UAV 的损毁位置。
        """
        plt.figure(figsize=(12, 9))

        if not target_df.empty:
            plt.scatter(
                target_df["x"],
                target_df["y"],
                marker="x",
                s=50,
                label="Targets",
                alpha=0.7,
            )

            for _, row in target_df.iterrows():
                plt.text(
                    row["x"],
                    row["y"],
                    str(int(row["target_id"])),
                    fontsize=8,
                    alpha=0.7,
                )

        for uav_id in selected_uav_ids:
            group = trajectory_df[trajectory_df["uav_id"] == uav_id].sort_values("time")

            if group.empty:
                continue

            plt.plot(
                group["x"],
                group["y"],
                linewidth=1.6,
                alpha=0.85,
                label=f"UAV {uav_id}",
            )

            start_row = group.iloc[0]
            end_row = group.iloc[-1]

            plt.scatter(start_row["x"], start_row["y"], marker="o", s=35)
            plt.scatter(end_row["x"], end_row["y"], marker="s", s=35)

            damaged_rows = group[group["is_damaged"].astype(str).str.lower() == "true"]

            if not damaged_rows.empty:
                first_damaged = damaged_rows.iloc[0]
                plt.scatter(
                    first_damaged["x"],
                    first_damaged["y"],
                    marker="X",
                    s=120,
                    label=f"UAV {uav_id} damaged",
                )

        event_annotations = self._extract_event_annotations(simulation_df)

        for event_time, event_types in event_annotations:
            plt.text(
                0.01,
                0.98 - 0.04 * len(event_types),
                f"t={event_time}: {', '.join(event_types)}",
                transform=plt.gca().transAxes,
                fontsize=9,
                verticalalignment="top",
            )

        plt.title("UAV Spatial Trajectories")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")

        if len(selected_uav_ids) <= 12:
            plt.legend(loc="best", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        return output_path

    def _save_uav_status_timeline_plot(
        self,
        trajectory_df: pd.DataFrame,
        selected_uav_ids: list[int],
        output_path: Path,
    ) -> Path:
        """
        绘制 UAV 状态时间线图。

        y 轴是 UAV ID，x 轴是仿真时间，颜色数值表示状态编码。
        """
        df = trajectory_df[
            trajectory_df["uav_id"].isin(selected_uav_ids)
        ].copy()

        if df.empty:
            raise MissionVisualizationError("No selected UAV trajectory data to plot.")

        df["status_code"] = df["status"].apply(self._status_to_code)

        pivot = df.pivot_table(
            index="uav_id",
            columns="time",
            values="status_code",
            aggfunc="last",
        ).sort_index()

        plt.figure(figsize=(12, 7))
        image = plt.imshow(
            pivot.values,
            aspect="auto",
            interpolation="nearest",
        )

        plt.colorbar(
            image,
            label="Status Code: available=0, assigned=1, en_route=2, striking=3, damaged=4",
        )

        plt.yticks(
            ticks=range(len(pivot.index)),
            labels=[str(int(value)) for value in pivot.index],
        )

        time_values = [float(value) for value in pivot.columns]
        tick_count = min(8, len(time_values))

        if tick_count > 0:
            tick_indices = [
                int(round(index * (len(time_values) - 1) / max(tick_count - 1, 1)))
                for index in range(tick_count)
            ]
            plt.xticks(
                ticks=tick_indices,
                labels=[f"{time_values[index]:.0f}" for index in tick_indices],
            )

        plt.title("UAV Status Timeline")
        plt.xlabel("Time")
        plt.ylabel("UAV ID")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        return output_path

    def _save_event_replanning_timeline_plot(
        self,
        simulation_df: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """
        绘制动态事件数量和重规划次数时间线。
        """
        df = simulation_df.copy()

        df["event_count"] = df["event_types"].apply(
            lambda value: len(self._parse_list(value))
        )

        if "replanning_count" not in df.columns:
            df["replanning_count"] = 0

        plt.figure(figsize=(11, 6))
        plt.plot(
            df["time"],
            df["event_count"],
            linewidth=2.0,
            marker="o",
            label="Event Count",
        )
        plt.plot(
            df["time"],
            df["replanning_count"],
            linewidth=2.0,
            marker="s",
            label="Replanning Count",
        )

        for _, row in df.iterrows():
            event_types = self._parse_list(row["event_types"])

            if event_types:
                plt.axvline(row["time"], linestyle="--", alpha=0.4)
                plt.text(
                    row["time"],
                    max(df["replanning_count"].max(), df["event_count"].max()) + 0.05,
                    ",".join(str(item) for item in event_types),
                    rotation=90,
                    fontsize=8,
                    verticalalignment="bottom",
                )

        plt.title("Dynamic Events and Replanning Timeline")
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        return output_path

    def _save_target_final_status_plot(
        self,
        target_df: pd.DataFrame,
        simulation_df: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """
        绘制目标最终状态空间图。

        根据最后一个时间步的：
        - destroyed_target_ids
        - disappeared_target_ids

        区分目标最终状态。
        """
        final_row = simulation_df.sort_values("time").iloc[-1]

        destroyed_ids = set(
            int(value) for value in self._parse_list(final_row["destroyed_target_ids"])
        )
        disappeared_ids = set(
            int(value) for value in self._parse_list(final_row["disappeared_target_ids"])
        )

        df = target_df.copy()

        def get_status(target_id: int) -> str:
            if int(target_id) in destroyed_ids:
                return "destroyed"
            if int(target_id) in disappeared_ids:
                return "disappeared"
            return "remaining"

        df["status"] = df["target_id"].apply(get_status)

        plt.figure(figsize=(10, 8))

        for status, group in df.groupby("status"):
            plt.scatter(
                group["x"],
                group["y"],
                s=70,
                label=status,
                alpha=0.8,
            )

            for _, row in group.iterrows():
                plt.text(
                    row["x"],
                    row["y"],
                    str(int(row["target_id"])),
                    fontsize=8,
                    alpha=0.8,
                )

        plt.title("Final Target Status Map")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        return output_path

    def _extract_event_annotations(
        self,
        simulation_df: pd.DataFrame,
    ) -> list[tuple[float, list[str]]]:
        """提取事件注释。"""
        annotations = []

        if "event_types" not in simulation_df.columns:
            return annotations

        for _, row in simulation_df.iterrows():
            event_types = self._parse_list(row["event_types"])

            if event_types:
                annotations.append(
                    (
                        float(row["time"]),
                        [str(value) for value in event_types],
                    )
                )

        return annotations

    @staticmethod
    def _status_to_code(status: str) -> int:
        """UAV 状态编码。"""
        mapping = {
            "available": 0,
            "assigned": 1,
            "en_route": 2,
            "striking": 3,
            "damaged": 4,
        }

        return mapping.get(str(status), -1)

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


def load_mission_visualization_config(
    config: dict[str, Any],
) -> MissionVisualizationConfig:
    """从项目配置中读取 MissionVisualizationConfig。"""
    visualization_config = MissionVisualizationConfig(
        uav_trajectory_csv_path=str(
            get_config_value(
                config,
                "mission_visualization.input.uav_trajectory_csv_path",
                default=get_config_value(
                    config,
                    "mission_simulator.output.uav_trajectory_csv_path",
                    default="outputs/simulation/uav_trajectory_log.csv",
                ),
            )
        ),
        simulation_log_csv_path=str(
            get_config_value(
                config,
                "mission_visualization.input.simulation_log_csv_path",
                default=get_config_value(
                    config,
                    "mission_simulator.output.simulation_log_csv_path",
                    default="outputs/simulation/mission_simulation_log.csv",
                ),
            )
        ),
        figure_dir=str(
            get_config_value(
                config,
                "mission_visualization.output.figure_dir",
                default="outputs/simulation/figures",
            )
        ),
        max_uavs_to_plot=int(
            get_config_value(
                config,
                "mission_visualization.max_uavs_to_plot",
                default=20,
            )
        ),
        only_assigned_uavs=bool(
            get_config_value(
                config,
                "mission_visualization.only_assigned_uavs",
                default=True,
            )
        ),
    )

    visualization_config.validate()
    return visualization_config


def build_mission_visualizer(
    config: dict[str, Any],
) -> MissionVisualizer:
    """构造 MissionVisualizer。"""
    return MissionVisualizer(
        config=load_mission_visualization_config(config)
    )