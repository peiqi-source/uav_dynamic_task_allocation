from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from uav_dynamic_task_allocation.utils.config import get_config_value, resolve_path


class DQNMetricsError(Exception):
    """DQN 训练指标保存、格式化和路径处理过程中的自定义错误。"""


@dataclass(frozen=True)
class DQNMetricsConfig:
    """
    DQN 训练指标配置。

    metrics_dir:
        指标文件保存目录。

    metrics_filename:
        episode 级别训练指标 CSV 文件名。

    overwrite:
        是否覆盖旧文件。调试阶段建议 True，避免旧实验数据混在一起。
        正式实验时可以改成 False，或者给不同实验设置不同文件名。

    save_jsonl:
        是否额外保存 jsonl 文件。CSV 适合画图，jsonl 适合保存复杂嵌套信息。
    """

    enabled: bool = True
    metrics_dir: str = "outputs/metrics"
    metrics_filename: str = "dqn_training_metrics.csv"
    overwrite: bool = True
    save_jsonl: bool = True
    jsonl_filename: str = "dqn_training_metrics.jsonl"

    def validate(self) -> None:
        """检查指标配置是否合法。"""
        if not self.metrics_dir:
            raise DQNMetricsError("metrics_dir must not be empty.")

        if not self.metrics_filename.endswith(".csv"):
            raise DQNMetricsError("metrics_filename should end with .csv.")

        if self.save_jsonl and not self.jsonl_filename.endswith(".jsonl"):
            raise DQNMetricsError("jsonl_filename should end with .jsonl.")


class DQNMetricsWriter:
    """
    DQN 训练指标写入器。

    该类负责把每个 episode 的训练结果写入 CSV / JSONL 文件。

    为什么不直接在 trainer.py 中写 csv？
    因为指标保存是一套独立逻辑，后面 train_dqn.py、evaluate_dqn.py、
    ablation experiments 都可能复用它。把它单独放在 metrics.py 中，
    可以让 trainer.py 保持简洁。
    """

    FIELDNAMES = [
        "episode",

        # StrikeOrder DQN 专用字段。
        # 对普通 DQN 来说这些字段可以为空。
        "cluster_id",
        "ordered_target_ids",
        "total_path_distance",

        # 通用 episode 训练指标。
        "total_reward",
        "num_steps",
        "num_updates",
        "mean_loss",
        "random_action_count",
        "greedy_action_count",
        "random_action_ratio",
        "invalid_or_no_action_count",
        "done",
        "is_best_episode",
        "epsilon_after_episode",
        "best_reward_so_far",
        "global_env_steps",
        "replay_buffer_size",
        "latest_checkpoint_path",
        "best_checkpoint_path",

        # 其他暂时没有放入固定列的信息。
        "extra",
    ]

    def __init__(
        self,
        config: DQNMetricsConfig,
        project_root: str | Path | None = None,
    ) -> None:
        config.validate()

        self.config = config
        self.project_root = Path(project_root) if project_root is not None else None

        self.metrics_dir = resolve_path(
            self.config.metrics_dir,
            project_root=self.project_root,
        )
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.metrics_dir / self.config.metrics_filename
        self.jsonl_path = self.metrics_dir / self.config.jsonl_filename

        self._prepare_output_files()

    def write_episode(self, episode_metrics: Mapping[str, Any]) -> None:
        """
        写入单个 episode 的指标。

        Args:
            episode_metrics:
                通常来自 EpisodeMetrics.to_dict()。
        """
        if not self.config.enabled:
            return

        row = self._prepare_csv_row(episode_metrics)

        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)

            if file.tell() == 0:
                writer.writeheader()

            writer.writerow(row)

        if self.config.save_jsonl:
            self._write_jsonl(episode_metrics)

    def write_many(self, metrics_list: list[Mapping[str, Any]]) -> None:
        """批量写入多个 episode 的指标。"""
        for metrics in metrics_list:
            self.write_episode(metrics)

    def _prepare_output_files(self) -> None:
        """
        根据 overwrite 配置处理已有文件。

        调试阶段 overwrite=True 更安全，可以避免上一次运行的结果混入本次实验。
        """
        if not self.config.enabled:
            return

        if self.config.overwrite:
            if self.csv_path.exists():
                self.csv_path.unlink()

            if self.config.save_jsonl and self.jsonl_path.exists():
                self.jsonl_path.unlink()

    def _prepare_csv_row(
        self,
        episode_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        把 episode_metrics 转换成 CSV 行。

        CSV 适合保存扁平字段。如果 episode_metrics 中有 FIELDNAMES 之外的字段，
        会统一打包进 extra 字段，避免信息丢失。
        """
        row: dict[str, Any] = {}

        for field_name in self.FIELDNAMES:
            if field_name == "extra":
                continue

            value = episode_metrics.get(field_name, "")
            row[field_name] = self._format_csv_value(value)

        extra = {
            key: value
            for key, value in episode_metrics.items()
            if key not in self.FIELDNAMES
        }

        row["extra"] = (
            json.dumps(extra, ensure_ascii=False, default=str)
            if extra
            else ""
        )

        return row

    def _write_jsonl(self, episode_metrics: Mapping[str, Any]) -> None:
        """
        写入 jsonl 格式指标。

        jsonl 每一行是一个 JSON 对象，比 CSV 更适合保存嵌套信息。
        """
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            json.dump(
                dict(episode_metrics),
                file,
                ensure_ascii=False,
                default=str,
            )
            file.write("\n")

    @staticmethod
    def _format_csv_value(value: Any) -> Any:
        """
        将 Python 对象转换成适合 CSV 保存的值。

        基础类型直接保存；
        list / dict / Path 等复杂对象转换成 JSON 字符串或普通字符串。
        """
        if value is None:
            return ""

        if isinstance(value, (str, int, float, bool)):
            return value

        return json.dumps(value, ensure_ascii=False, default=str)


def load_dqn_metrics_config(config: dict[str, Any]) -> DQNMetricsConfig:
    """
    从项目总配置中读取 DQN metrics 配置。

    这样指标保存路径和文件名由 YAML 控制，不写死在训练代码中。
    """
    metrics_config = DQNMetricsConfig(
        enabled=bool(
            get_config_value(
                config,
                "dqn.metrics.enabled",
                default=True,
            )
        ),
        metrics_dir=str(
            get_config_value(
                config,
                "dqn.metrics.metrics_dir",
                default="outputs/metrics",
            )
        ),
        metrics_filename=str(
            get_config_value(
                config,
                "dqn.metrics.metrics_filename",
                default="dqn_training_metrics.csv",
            )
        ),
        overwrite=bool(
            get_config_value(
                config,
                "dqn.metrics.overwrite",
                default=True,
            )
        ),
        save_jsonl=bool(
            get_config_value(
                config,
                "dqn.metrics.save_jsonl",
                default=True,
            )
        ),
        jsonl_filename=str(
            get_config_value(
                config,
                "dqn.metrics.jsonl_filename",
                default="dqn_training_metrics.jsonl",
            )
        ),
    )

    metrics_config.validate()
    return metrics_config