"""preprocessing 数据模块中的目标screening实现。"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from uav_dynamic_task_allocation.core.contracts import (
    AlgorithmMetadata,
    PipelineStage,
    ScreenedTargetSet,
    TargetScore,
)
from uav_dynamic_task_allocation.core.entities import Position, Target
from uav_dynamic_task_allocation.utils.config import get_config_value, resolve_path


class TargetScreeningError(Exception):
    """目标筛选、评分和结果导出过程中的自定义错误。"""


@dataclass(frozen=True)
class TargetScreeningConfig:
    """
    目标筛选配置。

    该模块对齐原代码 Target_Screen.py / Target_ScreenCore.py 的核心思想：
    1. 区分防御阵地和高价值目标；
    2. 根据防御阵地覆盖范围修正目标防御和重要性；
    3. 计算目标到参考点的距离和角度；
    4. 根据防御、重要性、距离和角度计算综合价值评分；
    5. 输出 ScreenedTargetSet，供后续 KMeans / PSO / PPO 分群使用。
    """

    # defense_post_type: 防御能力post类型。
    defense_post_type: int = 2
    # high_value_target_type: high数值目标类型。
    high_value_target_type: int = 1
    # defense_radius: 防御能力radius。
    defense_radius: float = 20.0
    # reference_position: reference位置坐标。
    reference_position: Position = field(
        default_factory=lambda: Position(x=0.0, y=0.0)
    )
    # use_defended_post_adjustment: usedefendedpostadjustment。
    use_defended_post_adjustment: bool = True

    # defense_decay_weight: 防御能力decay权重。
    defense_decay_weight: float = 0.1
    # significance_weight: 重要程度权重。
    significance_weight: float = 5.0
    # distance_decay_weight: distancedecay权重。
    distance_decay_weight: float = 32.0
    # angle_weight: angle权重。
    angle_weight: float = 1.0

    # defense_decay_base: 防御能力decay基础。
    defense_decay_base: float = 0.9
    # distance_decay_base: distancedecay基础。
    distance_decay_base: float = 0.9

    # selection_method: selectionmethod。
    selection_method: str = "all"
    # top_k: topk。
    top_k: int = 20
    # top_ratio: topratio。
    top_ratio: float = 0.5
    # threshold: threshold 数据。
    threshold: float = 0.5

    # debug_csv_path: 调试 CSV 输出路径。
    debug_csv_path: str = "outputs/intermediate/target_screening_scores.csv"

    def validate(self) -> None:
        """检查配置是否合法。"""
        if self.defense_radius <= 0:
            raise TargetScreeningError("defense_radius must be positive.")

        if self.defense_decay_base <= 0:
            raise TargetScreeningError("defense_decay_base must be positive.")

        if self.distance_decay_base <= 0:
            raise TargetScreeningError("distance_decay_base must be positive.")

        if self.selection_method not in {"all", "top_k", "top_ratio", "threshold"}:
            raise TargetScreeningError(
                "selection_method must be one of: all, top_k, top_ratio, threshold."
            )

        if self.top_k <= 0:
            raise TargetScreeningError("top_k must be positive.")

        if not 0 < self.top_ratio <= 1:
            raise TargetScreeningError("top_ratio must be in (0, 1].")


@dataclass
class TargetScreeningRecord:
    """
    单个目标的筛选计算记录。

    这个对象保存的是目标筛选阶段的中间信息。
    它不会替代 Target，而是作为 TargetScore 的补充信息保存。
    """

    # target: 目标。
    target: Target
    # raw_defense: raw防御能力。
    raw_defense: float
    # raw_significance: raw重要程度。
    raw_significance: float
    # adjusted_defense: adjusted防御能力。
    adjusted_defense: float
    # adjusted_significance: adjusted重要程度。
    adjusted_significance: float
    # distance_to_reference: distancetoreference。
    distance_to_reference: float
    # normalized_distance: normalizeddistance。
    normalized_distance: float
    # angle: angle 数据。
    angle: float
    # raw_score: raw评分。
    raw_score: float
    # normalized_score: normalized评分。
    normalized_score: float
    # affected_by_defense_posts: affectedby防御能力posts。
    affected_by_defense_posts: list[int] = field(default_factory=list)

    def to_target_score(self) -> TargetScore:
        """转换成 contracts.py 中定义的 TargetScore。"""
        return TargetScore(
            target_id=self.target.target_id,
            score=float(self.normalized_score),
            components={
                "raw_score": float(self.raw_score),
                "raw_defense": float(self.raw_defense),
                "raw_significance": float(self.raw_significance),
                "adjusted_defense": float(self.adjusted_defense),
                "adjusted_significance": float(self.adjusted_significance),
                "distance_to_reference": float(self.distance_to_reference),
                "normalized_distance": float(self.normalized_distance),
                "angle": float(self.angle),
            },
            metadata={
                "target_type": self._target_type_as_int(self.target),
                "affected_by_defense_posts": list(self.affected_by_defense_posts),
            },
        )

    @staticmethod
    def _target_type_as_int(target: Target) -> int:
        """尽量把 target_type 转成 int，兼容 int / float / Enum。"""
        target_type = target.target_type

        if hasattr(target_type, "value"):
            return int(target_type.value)

        return int(target_type)


class TargetScreener:
    """
    目标筛选器。

    该类是原 Target_Screen.py / Target_ScreenCore.py 的工程化版本。
    它不负责目标分群，也不负责 UAV 资源分配，只负责输出目标评分和筛选结果。
    """

    def __init__(self, config: TargetScreeningConfig) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            config: 配置对象，类型为 TargetScreeningConfig。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        config.validate()
        # config: 配置。
        self.config = config

    def screen(self, targets: list[Target]) -> ScreenedTargetSet:
        """
        对目标列表进行筛选和评分。

        Args:
            targets:
                全部战场目标。

        Returns:
            ScreenedTargetSet。
        """
        if not targets:
            raise TargetScreeningError("targets must not be empty.")

        adjusted_data = self._build_adjusted_target_data(targets)
        records = self._build_screening_records(targets, adjusted_data)

        selected_targets = self._select_targets(records)
        candidate_targets = [record.target for record in records]

        target_scores = {
            record.target.target_id: record.to_target_score()
            for record in records
        }

        screened_set = ScreenedTargetSet(
            all_targets=targets,
            candidate_targets=candidate_targets,
            selected_targets=selected_targets,
            target_scores=target_scores,
            algorithm_metadata=AlgorithmMetadata(
                algorithm_name="source_aligned_target_screening",
                algorithm_type="rule_based",
                stage=PipelineStage.TARGET_SCREENING,
                version="v1",
                config={
                    "defense_post_type": self.config.defense_post_type,
                    "high_value_target_type": self.config.high_value_target_type,
                    "defense_radius": self.config.defense_radius,
                    "selection_method": self.config.selection_method,
                },
                notes=(
                    "Reimplementation of Target_Screen.py / Target_ScreenCore.py "
                    "with structured output."
                ),
            ),
            metadata={
                "num_all_targets": len(targets),
                "num_candidate_targets": len(candidate_targets),
                "num_selected_targets": len(selected_targets),
                "selection_method": self.config.selection_method,
            },
        )

        screened_set.validate()
        return screened_set

    def write_debug_csv(
        self,
        screened_set: ScreenedTargetSet,
        output_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> Path:
        """
        将目标评分结果保存为 CSV，便于调试和后续报告分析。
        """
        path = resolve_path(
            output_path or self.config.debug_csv_path,
            project_root=project_root,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "target_id",
            "score",
            "raw_score",
            "target_type",
            "raw_defense",
            "raw_significance",
            "adjusted_defense",
            "adjusted_significance",
            "distance_to_reference",
            "normalized_distance",
            "angle",
            "affected_by_defense_posts",
            "is_selected",
        ]

        selected_ids = {target.target_id for target in screened_set.selected_targets}

        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for target_id, target_score in sorted(
                screened_set.target_scores.items(),
                key=lambda item: item[1].score,
                reverse=True,
            ):
                components = target_score.components
                metadata = target_score.metadata

                writer.writerow(
                    {
                        "target_id": target_id,
                        "score": target_score.score,
                        "raw_score": components.get("raw_score", ""),
                        "target_type": metadata.get("target_type", ""),
                        "raw_defense": components.get("raw_defense", ""),
                        "raw_significance": components.get("raw_significance", ""),
                        "adjusted_defense": components.get("adjusted_defense", ""),
                        "adjusted_significance": components.get(
                            "adjusted_significance", ""
                        ),
                        "distance_to_reference": components.get(
                            "distance_to_reference", ""
                        ),
                        "normalized_distance": components.get(
                            "normalized_distance", ""
                        ),
                        "angle": components.get("angle", ""),
                        "affected_by_defense_posts": metadata.get(
                            "affected_by_defense_posts", []
                        ),
                        "is_selected": target_id in selected_ids,
                    }
                )

        return path

    def _build_adjusted_target_data(
        self,
        targets: list[Target],
    ) -> dict[int, dict[str, Any]]:
        """
        构造修正后的目标属性。

        对齐原 Target_ScreenCore.py：
        - type=2 的目标作为防御阵地；
        - type=1 的目标作为高价值目标；
        - 如果高价值目标落入某防御阵地 R 范围内：
            防御阵地重要性增加 0.01 * 高价值目标重要性；
            高价值目标防御值受到防御阵地影响。

        原代码中 HVT defense 会被覆盖，存在顺序依赖。
        这里做了一个更鲁棒的改进：如果多个防御阵地影响同一 HVT，
        使用其中最大的 defense 作为 HVT 的 adjusted_defense。
        """
        data: dict[int, dict[str, Any]] = {}

        for target in targets:
            data[target.target_id] = {
                "target": target,
                "adjusted_defense": float(target.defense),
                "adjusted_significance": float(target.significance),
                "affected_by_defense_posts": [],
            }

        if not self.config.use_defended_post_adjustment:
            return data

        defense_posts = [
            target for target in targets
            if self._target_type_as_int(target) == self.config.defense_post_type
        ]

        high_value_targets = [
            target for target in targets
            if self._target_type_as_int(target) == self.config.high_value_target_type
        ]

        for defense_post in defense_posts:
            for hvt in high_value_targets:
                distance = self._distance(defense_post.position, hvt.position)

                if distance <= self.config.defense_radius:
                    data[defense_post.target_id]["adjusted_significance"] += (
                        0.01 * float(hvt.significance)
                    )

                    current_hvt_defense = data[hvt.target_id]["adjusted_defense"]
                    data[hvt.target_id]["adjusted_defense"] = max(
                        current_hvt_defense,
                        float(defense_post.defense),
                    )
                    data[hvt.target_id]["affected_by_defense_posts"].append(
                        defense_post.target_id
                    )

        return data

    def _build_screening_records(
        self,
        targets: list[Target],
        adjusted_data: dict[int, dict[str, Any]],
    ) -> list[TargetScreeningRecord]:
        """构造每个目标的筛选记录。"""
        distances = [
            self._distance(target.position, self.config.reference_position)
            for target in targets
        ]

        normalized_distances = self._normalize_distance_like_source(distances)

        raw_scores: list[float] = []
        temporary_records: list[dict[str, Any]] = []

        for target, normalized_distance, distance_to_reference in zip(
            targets,
            normalized_distances,
            distances,
            strict=False,
        ):
            adjusted = adjusted_data[target.target_id]

            angle = math.atan2(
                target.position.x - self.config.reference_position.x,
                target.position.y - self.config.reference_position.y,
            )

            adjusted_defense = float(adjusted["adjusted_defense"])
            adjusted_significance = float(adjusted["adjusted_significance"])

            raw_score = self._calculate_raw_score(
                adjusted_defense=adjusted_defense,
                adjusted_significance=adjusted_significance,
                normalized_distance=float(normalized_distance),
                angle=angle,
            )

            raw_scores.append(raw_score)

            temporary_records.append(
                {
                    "target": target,
                    "raw_defense": float(target.defense),
                    "raw_significance": float(target.significance),
                    "adjusted_defense": adjusted_defense,
                    "adjusted_significance": adjusted_significance,
                    "distance_to_reference": float(distance_to_reference),
                    "normalized_distance": float(normalized_distance),
                    "angle": float(angle),
                    "raw_score": float(raw_score),
                    "affected_by_defense_posts": list(
                        adjusted["affected_by_defense_posts"]
                    ),
                }
            )

        normalized_scores = self._minmax_scale(raw_scores)

        records: list[TargetScreeningRecord] = []

        for item, normalized_score in zip(
            temporary_records,
            normalized_scores,
            strict=False,
        ):
            records.append(
                TargetScreeningRecord(
                    target=item["target"],
                    raw_defense=item["raw_defense"],
                    raw_significance=item["raw_significance"],
                    adjusted_defense=item["adjusted_defense"],
                    adjusted_significance=item["adjusted_significance"],
                    distance_to_reference=item["distance_to_reference"],
                    normalized_distance=item["normalized_distance"],
                    angle=item["angle"],
                    raw_score=item["raw_score"],
                    normalized_score=float(normalized_score),
                    affected_by_defense_posts=item["affected_by_defense_posts"],
                )
            )

        return records

    def _calculate_raw_score(
        self,
        adjusted_defense: float,
        adjusted_significance: float,
        normalized_distance: float,
        angle: float,
    ) -> float:
        """
        计算原始综合评分。

        对齐原代码：
            reward = 0.1 * (0.9 ** defense)
                   + 10 * 0.5 * significance
                   + 80 * 0.4 * (0.9 ** distance_norm)
                   + abs(angle)

        当前将其中权重配置化，便于后续实验调整。
        """
        defense_term = (
            self.config.defense_decay_weight
            * (self.config.defense_decay_base ** adjusted_defense)
        )

        significance_term = (
            self.config.significance_weight * adjusted_significance
        )

        distance_term = (
            self.config.distance_decay_weight
            * (self.config.distance_decay_base ** normalized_distance)
        )

        angle_term = self.config.angle_weight * abs(angle)

        return float(defense_term + significance_term + distance_term + angle_term)

    def _select_targets(
        self,
        records: list[TargetScreeningRecord],
    ) -> list[Target]:
        """
        根据配置从评分结果中选择目标。

        当前默认 method=all，是为了对齐源代码流程：
        Target_Screen 之后还会进入 Kmeans_step1 做摧毁目标集划分。
        """
        sorted_records = sorted(
            records,
            key=lambda record: record.normalized_score,
            reverse=True,
        )

        if self.config.selection_method == "all":
            return [record.target for record in sorted_records]

        if self.config.selection_method == "top_k":
            return [
                record.target
                for record in sorted_records[: self.config.top_k]
            ]

        if self.config.selection_method == "top_ratio":
            k = max(1, int(math.ceil(len(sorted_records) * self.config.top_ratio)))
            return [
                record.target
                for record in sorted_records[:k]
            ]

        if self.config.selection_method == "threshold":
            return [
                record.target
                for record in sorted_records
                if record.normalized_score >= self.config.threshold
            ]

        raise TargetScreeningError(
            f"Unsupported selection_method: {self.config.selection_method}"
        )

    def _normalize_distance_like_source(
        self,
        distances: list[float],
    ) -> list[float]:
        """
        对齐原 Target_ScreenCore.py 的距离归一化思路。

        原代码：
            distanceUAV_list_normalized = [val / distanceUAVmin for val in distanceUAV_list]
            distanceUAV_list_normalized = minmax_scale(distanceUAV_list_normalized)

        这里避免 sklearn 依赖，使用本地 min-max 实现。
        """
        if not distances:
            return []

        positive_distances = [distance for distance in distances if distance > 0]
        min_distance = min(positive_distances) if positive_distances else 1.0

        ratio_values = [
            distance / max(min_distance, 1e-8)
            for distance in distances
        ]

        return self._minmax_scale(ratio_values)

    @staticmethod
    def _minmax_scale(values: list[float]) -> list[float]:
        """简单 min-max 归一化。"""
        if not values:
            return []

        min_value = min(values)
        max_value = max(values)

        if abs(max_value - min_value) < 1e-12:
            return [0.0 for _ in values]

        return [
            (value - min_value) / (max_value - min_value)
            for value in values
        ]

    @staticmethod
    def _distance(position_a: Position, position_b: Position) -> float:
        """计算欧氏距离。"""
        return math.sqrt(
            (position_a.x - position_b.x) ** 2
            + (position_a.y - position_b.y) ** 2
        )

    @staticmethod
    def _target_type_as_int(target: Target) -> int:
        """尽量把 target_type 转成 int，兼容 int / float / Enum。"""
        target_type = target.target_type

        if hasattr(target_type, "value"):
            return int(target_type.value)

        return int(target_type)


def load_target_screening_config(
    config: dict[str, Any],
) -> TargetScreeningConfig:
    """从项目总配置中读取目标筛选配置。"""
    reference_position_raw = get_config_value(
        config,
        "target_screening.reference_position",
        default=[0.0, 0.0],
    )

    screening_config = TargetScreeningConfig(
        defense_post_type=int(
            get_config_value(
                config,
                "target_screening.defense_post_type",
                default=2,
            )
        ),
        high_value_target_type=int(
            get_config_value(
                config,
                "target_screening.high_value_target_type",
                default=1,
            )
        ),
        defense_radius=float(
            get_config_value(
                config,
                "target_screening.defense_radius",
                default=20.0,
            )
        ),
        reference_position=Position(
            x=float(reference_position_raw[0]),
            y=float(reference_position_raw[1]),
        ),
        use_defended_post_adjustment=bool(
            get_config_value(
                config,
                "target_screening.use_defended_post_adjustment",
                default=True,
            )
        ),
        defense_decay_weight=float(
            get_config_value(
                config,
                "target_screening.score.defense_decay_weight",
                default=0.1,
            )
        ),
        significance_weight=float(
            get_config_value(
                config,
                "target_screening.score.significance_weight",
                default=5.0,
            )
        ),
        distance_decay_weight=float(
            get_config_value(
                config,
                "target_screening.score.distance_decay_weight",
                default=32.0,
            )
        ),
        angle_weight=float(
            get_config_value(
                config,
                "target_screening.score.angle_weight",
                default=1.0,
            )
        ),
        defense_decay_base=float(
            get_config_value(
                config,
                "target_screening.score.defense_decay_base",
                default=0.9,
            )
        ),
        distance_decay_base=float(
            get_config_value(
                config,
                "target_screening.score.distance_decay_base",
                default=0.9,
            )
        ),
        selection_method=str(
            get_config_value(
                config,
                "target_screening.selection.method",
                default="all",
            )
        ),
        top_k=int(
            get_config_value(
                config,
                "target_screening.selection.top_k",
                default=20,
            )
        ),
        top_ratio=float(
            get_config_value(
                config,
                "target_screening.selection.top_ratio",
                default=0.5,
            )
        ),
        threshold=float(
            get_config_value(
                config,
                "target_screening.selection.threshold",
                default=0.5,
            )
        ),
        debug_csv_path=str(
            get_config_value(
                config,
                "target_screening.output.debug_csv_path",
                default="outputs/intermediate/target_screening_scores.csv",
            )
        ),
    )

    screening_config.validate()
    return screening_config
