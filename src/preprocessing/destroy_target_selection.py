from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import joblib

import numpy as np

from uav_dynamic_task_allocation.core.contracts import (
    AlgorithmMetadata,
    PipelineStage,
    ScreenedTargetSet,
    TargetScore,
)
from uav_dynamic_task_allocation.core.entities import Target
from uav_dynamic_task_allocation.utils.config import get_config_value, resolve_path


class DestroyTargetSelectionError(Exception):
    """摧毁目标集选择过程中的自定义错误。"""


@dataclass(frozen=True)
class DestroyTargetSelectionConfig:
    """
    摧毁目标集选择配置。

    该配置支持多种算法：
    - weighted_kmeans：对齐源代码 main.py 实际使用的 KMeans_step1；
    - random_forest：对齐论文中的随机森林摧毁目标集决策思想；
    - top_k：规则法 baseline；
    - threshold：规则法 baseline。

    所有方法都必须输出 DestroyTargetSelectionResult。
    这样后续 target_clustering.py、resource_allocation.py、
    StrikeOrder DQN 和 MissionSimulator 都不需要关心前面到底用了哪种算法。
    """

    method: str = "weighted_kmeans"

    # weighted_kmeans 配置
    num_clusters: int = 2
    max_iter: int = 300
    tolerance: float = 0.0004
    random_seed: int = 42

    normalized_distance_weight: float = 0.4
    score_weight: float = 0.6

    choose_cluster_by: str = "highest_score_cluster"

    # random forest 配置
    rf_model_path: str = "checkpoints/random_forest/destroy_target_rf.joblib"
    rf_probability_threshold: float = 0.5
    rf_fallback_method: str = "top_k"
    rf_fallback_top_k: int = 5
    rf_feature_names: tuple[str, ...] = (
        "score",
        "normalized_distance",
        "angle",
        "adjusted_defense",
        "adjusted_significance",
    )

    # rule baseline 配置
    rule_top_k: int = 8
    rule_threshold: float = 0.5

    debug_csv_path: str = "outputs/intermediate/destroy_target_selection.csv"

    def validate(self) -> None:
        """检查配置是否合法。"""
        supported_methods = {
            "weighted_kmeans",
            "random_forest",
            "top_k",
            "threshold",
        }

        if self.method not in supported_methods:
            raise DestroyTargetSelectionError(
                f"Unsupported method={self.method}. "
                f"Supported methods: {sorted(supported_methods)}"
            )

        if self.num_clusters < 2:
            raise DestroyTargetSelectionError("num_clusters must be at least 2.")

        if self.max_iter <= 0:
            raise DestroyTargetSelectionError("max_iter must be positive.")

        if self.tolerance <= 0:
            raise DestroyTargetSelectionError("tolerance must be positive.")

        if self.normalized_distance_weight < 0 or self.score_weight < 0:
            raise DestroyTargetSelectionError("feature weights must be non-negative.")

        if self.normalized_distance_weight + self.score_weight <= 0:
            raise DestroyTargetSelectionError(
                "At least one feature weight must be positive."
            )

        if self.choose_cluster_by != "highest_score_cluster":
            raise DestroyTargetSelectionError(
                "Only choose_cluster_by='highest_score_cluster' is supported currently."
            )

        if not 0.0 <= self.rf_probability_threshold <= 1.0:
            raise DestroyTargetSelectionError(
                "rf_probability_threshold must be in [0, 1]."
            )

        if self.rf_fallback_method not in {"top_k", "threshold"}:
            raise DestroyTargetSelectionError(
                "rf_fallback_method must be 'top_k' or 'threshold'."
            )

        if self.rf_fallback_top_k <= 0:
            raise DestroyTargetSelectionError("rf_fallback_top_k must be positive.")

        if self.rule_top_k <= 0:
            raise DestroyTargetSelectionError("rule_top_k must be positive.")

        if not 0.0 <= self.rule_threshold <= 1.0:
            raise DestroyTargetSelectionError("rule_threshold must be in [0, 1].")


@dataclass
class DestroyTargetSelectionRecord:
    """
    单个目标的摧毁目标集选择记录。

    这个记录用于保存：
    - 目标评分；
    - 距离特征；
    - KMeans cluster label；
    - 是否被选入摧毁目标集。
    """

    target: Target
    score: float
    normalized_distance: float
    cluster_label: int
    is_destroy_target: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DestroyTargetSelectionResult:
    """
    摧毁目标集选择结果。

    destroy_targets:
        被选入摧毁目标集的目标。

    non_destroy_targets:
        没有被选入摧毁目标集的目标。

    records:
        每个目标的选择记录。

    selected_cluster_label:
        被判定为摧毁目标集的 KMeans cluster 编号。

    to_screened_target_set():
        将结果转换回 ScreenedTargetSet，使后续 target_clustering.py
        可以继续读取 screened_set.selected_targets。
    """

    source_screened_set: ScreenedTargetSet
    destroy_targets: list[Target]
    non_destroy_targets: list[Target]
    records: list[DestroyTargetSelectionRecord]
    selected_cluster_label: int
    algorithm_metadata: AlgorithmMetadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_screened_target_set(self) -> ScreenedTargetSet:
        """
        转换成新的 ScreenedTargetSet。

        这里 selected_targets 会被更新为 destroy_targets。
        candidate_targets 保留原候选目标。
        target_scores 保留原评分结果。
        """
        selected_set = ScreenedTargetSet(
            all_targets=self.source_screened_set.all_targets,
            candidate_targets=self.source_screened_set.candidate_targets,
            selected_targets=self.destroy_targets,
            target_scores=self.source_screened_set.target_scores,
            algorithm_metadata=self.algorithm_metadata,
            metadata={
                **self.source_screened_set.metadata,
                **self.metadata,
                "num_destroy_targets": len(self.destroy_targets),
                "num_non_destroy_targets": len(self.non_destroy_targets),
                "selected_cluster_label": self.selected_cluster_label,
                "previous_stage": "target_screening",
                "current_stage": "destroy_target_selection",
            },
        )
        selected_set.validate()
        return selected_set


class DestroyTargetSelector:
    """
    摧毁目标集选择器。

    它输入 TargetScreener 输出的 ScreenedTargetSet，
    输出 DestroyTargetSelectionResult。

    该模块是原 Kmeans_step1.py 的工程化版本。
    它不会负责后续目标分群；它只把目标分成：
    - destroy_targets；
    - non_destroy_targets。
    """

    def __init__(self, config: DestroyTargetSelectionConfig) -> None:
        config.validate()
        self.config = config

    def select(
            self,
            screened_set: ScreenedTargetSet,
    ) -> DestroyTargetSelectionResult:
        """
        选择摧毁目标集。

        该函数是摧毁目标集决策阶段的统一入口。

        根据 config.method 自动调用不同算法：
        - weighted_kmeans：对齐源代码 main.py 中实际使用的 KMeans_step1；
        - random_forest：对齐论文中的随机森林摧毁目标集决策思想；
        - top_k：规则法 baseline；
        - threshold：规则法 baseline。

        这样设计的目的是让摧毁目标集决策模块具有良好的算法兼容性。
        后续无论替换为 Random Forest、XGBoost、PPO、专家规则还是多目标优化，
        只要最终输出 DestroyTargetSelectionResult，后面的目标分群、资源分配、
        StrikeOrder DQN 和 MissionSimulator 都不需要修改。
        """
        screened_set.validate()

        if not screened_set.candidate_targets:
            raise DestroyTargetSelectionError(
                "screened_set.candidate_targets must not be empty."
            )

        if self.config.method == "weighted_kmeans":
            return self._select_by_weighted_kmeans(screened_set)

        if self.config.method == "random_forest":
            return self._select_by_random_forest(screened_set)

        if self.config.method == "top_k":
            return self._select_by_top_k(screened_set)

        if self.config.method == "threshold":
            return self._select_by_threshold(screened_set)

        raise DestroyTargetSelectionError(
            f"Unsupported destroy target selection method: {self.config.method}"
        )

    def _select_by_weighted_kmeans(
            self,
            screened_set: ScreenedTargetSet,
    ) -> DestroyTargetSelectionResult:
        """
        使用加权 KMeans 选择摧毁目标集。

        该方法对应原代码 Kmeans_step1.py / Kmeans_step1Core.py。
        主要思想是：
        1. 根据目标 score 和 normalized_distance 构造特征；
        2. 使用加权 KMeans 聚成若干类；
        3. 找到包含最高 score 目标的 cluster；
        4. 将该 cluster 作为摧毁目标集。
        """
        targets = screened_set.candidate_targets

        if len(targets) < self.config.num_clusters:
            raise DestroyTargetSelectionError(
                "Number of candidate targets must be >= num_clusters: "
                f"num_targets={len(targets)}, num_clusters={self.config.num_clusters}"
            )

        features, target_ids = self._build_feature_matrix(screened_set)

        labels, centers, num_iter = self._weighted_kmeans(features)

        selected_cluster_label = self._choose_destroy_cluster(
            labels=labels,
            screened_set=screened_set,
            target_ids=target_ids,
        )

        records: list[DestroyTargetSelectionRecord] = []
        destroy_targets: list[Target] = []
        non_destroy_targets: list[Target] = []

        target_by_id = {
            target.target_id: target
            for target in targets
        }

        for row_index, target_id in enumerate(target_ids):
            target = target_by_id[target_id]
            target_score = screened_set.target_scores[target_id]

            cluster_label = int(labels[row_index])
            is_destroy_target = cluster_label == selected_cluster_label

            normalized_distance = self._get_normalized_distance(target_score)
            score = float(target_score.score)

            record = DestroyTargetSelectionRecord(
                target=target,
                score=score,
                normalized_distance=normalized_distance,
                cluster_label=cluster_label,
                is_destroy_target=is_destroy_target,
                metadata={
                    "method": "weighted_kmeans",
                    "feature_vector": features[row_index].tolist(),
                },
            )
            records.append(record)

            if is_destroy_target:
                destroy_targets.append(target)
            else:
                non_destroy_targets.append(target)

        algorithm_metadata = AlgorithmMetadata(
            algorithm_name="source_aligned_kmeans_step1",
            algorithm_type="weighted_kmeans",
            stage=PipelineStage.TARGET_SCREENING,
            version="v1",
            config={
                "num_clusters": self.config.num_clusters,
                "max_iter": self.config.max_iter,
                "tolerance": self.config.tolerance,
                "normalized_distance_weight": self.config.normalized_distance_weight,
                "score_weight": self.config.score_weight,
                "choose_cluster_by": self.config.choose_cluster_by,
            },
            notes=(
                "Reimplementation of Kmeans_step1.py / Kmeans_step1Core.py. "
                "The cluster containing the highest-score target is selected "
                "as the destroy target set."
            ),
        )

        result = DestroyTargetSelectionResult(
            source_screened_set=screened_set,
            destroy_targets=destroy_targets,
            non_destroy_targets=non_destroy_targets,
            records=records,
            selected_cluster_label=selected_cluster_label,
            algorithm_metadata=algorithm_metadata,
            metadata={
                "method": "weighted_kmeans",
                "num_candidate_targets": len(targets),
                "num_destroy_targets": len(destroy_targets),
                "num_non_destroy_targets": len(non_destroy_targets),
                "cluster_centers": centers.tolist(),
                "num_iter": num_iter,
            },
        )

        return result

    def _select_by_random_forest(
            self,
            screened_set: ScreenedTargetSet,
    ) -> DestroyTargetSelectionResult:
        """
        使用随机森林模型进行摧毁目标集决策。

        该方法对应论文中的“基于随机森林算法的摧毁目标集决策”思想。
        它不重新计算目标评分，而是使用 TargetScreener 已经生成的目标特征。

        输入：
            ScreenedTargetSet

        输出：
            DestroyTargetSelectionResult

        注意：
            Random Forest 是监督学习算法，需要训练好的模型。
            如果模型不存在，应先运行 scripts/train_destroy_target_rf.py。
        """
        model_path = resolve_path(self.config.rf_model_path)

        if not model_path.exists():
            raise DestroyTargetSelectionError(
                f"Random forest model not found: {model_path}. "
                "Please train it first with scripts/train_destroy_target_rf.py, "
                "or set destroy_target_selection.method=weighted_kmeans."
            )

        try:
            model = joblib.load(model_path)
        except Exception as exc:
            return self._select_by_random_forest_fallback(
                screened_set=screened_set,
                reason=f"model_load_failed: {exc}",
            )

        features, target_ids = self._build_rf_feature_matrix(screened_set)

        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)

                if probabilities.shape[1] < 2:
                    destroy_probabilities = probabilities[:, 0]
                else:
                    destroy_probabilities = probabilities[:, 1]

                labels = (
                        destroy_probabilities >= self.config.rf_probability_threshold
                ).astype(int)
            else:
                labels = model.predict(features).astype(int)
                destroy_probabilities = labels.astype(float)
        except Exception as exc:
            return self._select_by_random_forest_fallback(
                screened_set=screened_set,
                reason=f"model_inference_failed: {exc}",
            )

        destroy_ids = [
            target_id
            for target_id, label in zip(target_ids, labels, strict=False)
            if int(label) == 1
        ]

        # 防止模型极端情况下一个目标都不选，导致后续流程断掉。
        if not destroy_ids:
            destroy_ids = self._fallback_destroy_ids(
                screened_set=screened_set,
                method=self.config.rf_fallback_method,
                top_k=self.config.rf_fallback_top_k,
                threshold=self.config.rule_threshold,
            )

        target_by_id = {
            target.target_id: target
            for target in screened_set.candidate_targets
        }

        destroy_targets = [
            target_by_id[target_id]
            for target_id in destroy_ids
            if target_id in target_by_id
        ]

        destroy_id_set = {target.target_id for target in destroy_targets}

        non_destroy_targets = [
            target
            for target in screened_set.candidate_targets
            if target.target_id not in destroy_id_set
        ]

        records: list[DestroyTargetSelectionRecord] = []

        for index, target_id in enumerate(target_ids):
            target = target_by_id[target_id]
            target_score = screened_set.target_scores[target_id]

            record = DestroyTargetSelectionRecord(
                target=target,
                score=float(target_score.score),
                normalized_distance=self._get_normalized_distance(target_score),
                cluster_label=int(labels[index]),
                is_destroy_target=target_id in destroy_id_set,
                metadata={
                    "method": "random_forest",
                    "destroy_probability": float(destroy_probabilities[index]),
                    "feature_names": list(self.config.rf_feature_names),
                    "feature_vector": features[index].tolist(),
                },
            )
            records.append(record)

        algorithm_metadata = AlgorithmMetadata(
            algorithm_name="random_forest_destroy_target_selector",
            algorithm_type="random_forest",
            stage=PipelineStage.TARGET_SCREENING,
            version="v1",
            config={
                "model_path": str(model_path),
                "probability_threshold": self.config.rf_probability_threshold,
                "feature_names": list(self.config.rf_feature_names),
                "fallback_method": self.config.rf_fallback_method,
            },
            notes=(
                "Random Forest based destroy target decision. "
                "This module implements the supervised-learning branch mentioned "
                "in the paper and provides a pluggable alternative to weighted_kmeans."
            ),
        )

        return DestroyTargetSelectionResult(
            source_screened_set=screened_set,
            destroy_targets=destroy_targets,
            non_destroy_targets=non_destroy_targets,
            records=records,
            selected_cluster_label=1,
            algorithm_metadata=algorithm_metadata,
            metadata={
                "method": "random_forest",
                "num_candidate_targets": len(screened_set.candidate_targets),
                "num_destroy_targets": len(destroy_targets),
                "num_non_destroy_targets": len(non_destroy_targets),
                "model_path": str(model_path),
            },
        )

    def _select_by_random_forest_fallback(
            self,
            screened_set: ScreenedTargetSet,
            reason: str,
    ) -> DestroyTargetSelectionResult:
        """
        Random Forest 妯″瀷涓嶅彲鐢ㄦ椂鐨勫伐绋嬪寲 fallback銆?

        杩欑鎯呭喌甯歌浜庢ā鍨?joblib 鏂囦欢涓?scikit-learn 鐗堟湰涓嶅吋瀹广€?
        fallback 鍙敤浜庝繚璇佹祦绋嬮棴鐜紝涓嶅簲褰撲綔璁烘枃涓?Random Forest
        绠楁硶鐨勬寮忓鐜扮粨鏋溿€?
        """
        destroy_ids = self._fallback_destroy_ids(
            screened_set=screened_set,
            method=self.config.rf_fallback_method,
            top_k=self.config.rf_fallback_top_k,
            threshold=self.config.rule_threshold,
        )

        target_by_id = {
            target.target_id: target
            for target in screened_set.candidate_targets
        }

        destroy_targets = [
            target_by_id[target_id]
            for target_id in destroy_ids
            if target_id in target_by_id
        ]

        result = self._build_rule_based_result(
            screened_set=screened_set,
            destroy_targets=destroy_targets,
            method_name=f"random_forest_fallback_{self.config.rf_fallback_method}",
        )

        result.metadata.update(
            {
                "requested_method": "random_forest",
                "fallback_reason": reason,
                "fallback_method": self.config.rf_fallback_method,
                "rf_model_path": self.config.rf_model_path,
            }
        )

        return result

    def _select_by_top_k(
            self,
            screened_set: ScreenedTargetSet,
    ) -> DestroyTargetSelectionResult:
        """
        规则 baseline：选择评分最高的 top_k 个目标作为摧毁目标集。
        """
        top_k = min(self.config.rule_top_k, len(screened_set.candidate_targets))

        sorted_targets = sorted(
            screened_set.candidate_targets,
            key=lambda target: screened_set.target_scores[target.target_id].score,
            reverse=True,
        )

        destroy_targets = sorted_targets[:top_k]
        return self._build_rule_based_result(
            screened_set=screened_set,
            destroy_targets=destroy_targets,
            method_name="top_k",
        )

    def _select_by_threshold(
            self,
            screened_set: ScreenedTargetSet,
    ) -> DestroyTargetSelectionResult:
        """
        规则 baseline：选择 score >= threshold 的目标作为摧毁目标集。
        """
        destroy_targets = [
            target
            for target in screened_set.candidate_targets
            if screened_set.target_scores[target.target_id].score
               >= self.config.rule_threshold
        ]

        if not destroy_targets:
            destroy_targets = sorted(
                screened_set.candidate_targets,
                key=lambda target: screened_set.target_scores[target.target_id].score,
                reverse=True,
            )[:1]

        return self._build_rule_based_result(
            screened_set=screened_set,
            destroy_targets=destroy_targets,
            method_name="threshold",
        )

    def _build_rule_based_result(
            self,
            screened_set: ScreenedTargetSet,
            destroy_targets: list[Target],
            method_name: str,
    ) -> DestroyTargetSelectionResult:
        """
        构造规则法选择结果。
        """
        destroy_id_set = {target.target_id for target in destroy_targets}

        non_destroy_targets = [
            target
            for target in screened_set.candidate_targets
            if target.target_id not in destroy_id_set
        ]

        records: list[DestroyTargetSelectionRecord] = []

        for target in screened_set.candidate_targets:
            target_score = screened_set.target_scores[target.target_id]

            records.append(
                DestroyTargetSelectionRecord(
                    target=target,
                    score=float(target_score.score),
                    normalized_distance=self._get_normalized_distance(target_score),
                    cluster_label=1 if target.target_id in destroy_id_set else 0,
                    is_destroy_target=target.target_id in destroy_id_set,
                    metadata={
                        "method": method_name,
                    },
                )
            )

        algorithm_metadata = AlgorithmMetadata(
            algorithm_name=f"rule_based_{method_name}_destroy_selector",
            algorithm_type="rule_based",
            stage=PipelineStage.TARGET_SCREENING,
            version="v1",
            config={
                "method": method_name,
                "top_k": self.config.rule_top_k,
                "threshold": self.config.rule_threshold,
            },
            notes="Rule-based destroy target selection baseline.",
        )

        return DestroyTargetSelectionResult(
            source_screened_set=screened_set,
            destroy_targets=destroy_targets,
            non_destroy_targets=non_destroy_targets,
            records=records,
            selected_cluster_label=1,
            algorithm_metadata=algorithm_metadata,
            metadata={
                "method": method_name,
                "num_candidate_targets": len(screened_set.candidate_targets),
                "num_destroy_targets": len(destroy_targets),
                "num_non_destroy_targets": len(non_destroy_targets),
            },
        )

    def _build_rf_feature_matrix(
            self,
            screened_set: ScreenedTargetSet,
    ) -> tuple[np.ndarray, list[int]]:
        """
        构造 Random Forest 使用的特征矩阵。

        feature_names 来自配置：
            score
            normalized_distance
            angle
            adjusted_defense
            adjusted_significance
            raw_defense
            raw_significance

        这样以后可以非常方便地增删 RF 特征。
        """
        features: list[list[float]] = []
        target_ids: list[int] = []

        for target in screened_set.candidate_targets:
            target_id = target.target_id
            target_score = screened_set.target_scores[target_id]

            row = [
                self._get_feature_value(target_score, feature_name)
                for feature_name in self.config.rf_feature_names
            ]

            features.append(row)
            target_ids.append(target_id)

        return np.array(features, dtype=np.float64), target_ids

    def _get_feature_value(
            self,
            target_score: TargetScore,
            feature_name: str,
    ) -> float:
        """
        从 TargetScore 中读取 RF 特征值。
        """
        if feature_name == "score":
            return float(target_score.score)

        if feature_name in target_score.components:
            return float(target_score.components[feature_name])

        if feature_name in target_score.metadata:
            return float(target_score.metadata[feature_name])

        raise DestroyTargetSelectionError(
            f"Feature '{feature_name}' not found in TargetScore. "
            f"Available components={list(target_score.components.keys())}, "
            f"metadata={list(target_score.metadata.keys())}"
        )

    def _fallback_destroy_ids(
            self,
            screened_set: ScreenedTargetSet,
            method: str,
            top_k: int,
            threshold: float,
    ) -> list[int]:
        """
        当 RF 没有选出任何目标时，使用 fallback 防止流程中断。
        """
        sorted_targets = sorted(
            screened_set.candidate_targets,
            key=lambda target: screened_set.target_scores[target.target_id].score,
            reverse=True,
        )

        if method == "top_k":
            return [
                target.target_id
                for target in sorted_targets[: min(top_k, len(sorted_targets))]
            ]

        if method == "threshold":
            selected = [
                target.target_id
                for target in sorted_targets
                if screened_set.target_scores[target.target_id].score >= threshold
            ]

            if selected:
                return selected

            return [sorted_targets[0].target_id]

        raise DestroyTargetSelectionError(f"Unsupported fallback method: {method}")

    def write_debug_csv(
        self,
        result: DestroyTargetSelectionResult,
        output_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> Path:
        """保存摧毁目标集选择结果，便于检查和报告分析。"""
        path = resolve_path(
            output_path or self.config.debug_csv_path,
            project_root=project_root,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "target_id",
            "score",
            "normalized_distance",
            "cluster_label",
            "is_destroy_target",
            "target_type",
            "defense",
            "significance",
            "x",
            "y",
        ]

        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for record in sorted(
                result.records,
                key=lambda item: item.score,
                reverse=True,
            ):
                target = record.target

                writer.writerow(
                    {
                        "target_id": target.target_id,
                        "score": record.score,
                        "normalized_distance": record.normalized_distance,
                        "cluster_label": record.cluster_label,
                        "is_destroy_target": record.is_destroy_target,
                        "target_type": self._target_type_as_int(target),
                        "defense": target.defense,
                        "significance": target.significance,
                        "x": target.position.x,
                        "y": target.position.y,
                    }
                )

        return path

    def _build_feature_matrix(
        self,
        screened_set: ScreenedTargetSet,
    ) -> tuple[np.ndarray, list[int]]:
        """
        构造 KMeans 特征矩阵。

        当前使用两个特征：
        1. normalized_distance
        2. score

        这对应原 Kmeans_step1Core.py 中对两个特征分别使用 0.4 和 0.6 权重。
        """
        features: list[list[float]] = []
        target_ids: list[int] = []

        for target in screened_set.candidate_targets:
            target_id = target.target_id

            if target_id not in screened_set.target_scores:
                raise DestroyTargetSelectionError(
                    f"Target score not found for target_id={target_id}"
                )

            target_score = screened_set.target_scores[target_id]

            normalized_distance = self._get_normalized_distance(target_score)
            score = float(target_score.score)

            features.append([normalized_distance, score])
            target_ids.append(target_id)

        return np.array(features, dtype=np.float64), target_ids

    def _weighted_kmeans(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        加权 KMeans。

        与原 Kmeans_step1Core.py 保持相同思想：
        - assignment 阶段使用加权 L1 距离；
        - update 阶段使用 cluster 均值；
        - RSS 变化小于 tolerance 后停止。
        """
        num_samples, dim = features.shape

        if num_samples < self.config.num_clusters:
            raise DestroyTargetSelectionError(
                "num_samples must be >= num_clusters."
            )

        rng = np.random.default_rng(self.config.random_seed)

        initial_indices = rng.choice(
            num_samples,
            size=self.config.num_clusters,
            replace=False,
        )

        centers = features[initial_indices].copy()
        labels = np.zeros(num_samples, dtype=np.int64)

        weights = np.array(
            [
                self.config.normalized_distance_weight,
                self.config.score_weight,
            ],
            dtype=np.float64,
        )

        if dim != len(weights):
            raise DestroyTargetSelectionError(
                f"Feature dimension mismatch: dim={dim}, weights={len(weights)}"
            )

        previous_error = float("inf")

        for iteration in range(1, self.config.max_iter + 1):
            # assignment
            for sample_index in range(num_samples):
                distances = np.sum(
                    np.abs(features[sample_index] - centers) * weights,
                    axis=1,
                )
                labels[sample_index] = int(np.argmin(distances))

            # update
            new_centers = centers.copy()
            for cluster_index in range(self.config.num_clusters):
                cluster_points = features[labels == cluster_index]

                if len(cluster_points) > 0:
                    new_centers[cluster_index] = cluster_points.mean(axis=0)
                else:
                    # 空 cluster 时，重新随机选择一个样本作为中心。
                    replacement_index = int(rng.integers(0, num_samples))
                    new_centers[cluster_index] = features[replacement_index]

            centers = new_centers

            # weighted RSS
            error = 0.0
            for sample_index in range(num_samples):
                diff = features[sample_index] - centers[labels[sample_index]]
                error += float(np.sum((diff * weights) ** 2))
            error /= num_samples

            if abs(previous_error - error) < self.config.tolerance:
                return labels, centers, iteration

            previous_error = error

        return labels, centers, self.config.max_iter

    def _choose_destroy_cluster(
        self,
        labels: np.ndarray,
        screened_set: ScreenedTargetSet,
        target_ids: list[int],
    ) -> int:
        """
        选择摧毁目标集对应的 cluster。

        当前策略：
        找到 score 最高的目标，看它属于哪个 cluster；
        这个 cluster 就是摧毁目标集。

        这对应原 Kmeans_step1.py 中：
        找到包含最大综合价值目标的 cluster。
        """
        best_target_id = None
        best_score = -float("inf")

        for target_id in target_ids:
            score = float(screened_set.target_scores[target_id].score)

            if score > best_score:
                best_score = score
                best_target_id = target_id

        if best_target_id is None:
            raise DestroyTargetSelectionError("Could not find highest-score target.")

        best_index = target_ids.index(best_target_id)
        return int(labels[best_index])

    @staticmethod
    def _get_normalized_distance(target_score: TargetScore) -> float:
        """从 TargetScore 中读取 normalized_distance。"""
        value = target_score.components.get("normalized_distance", None)

        if value is None:
            return 0.0

        return float(value)

    @staticmethod
    def _target_type_as_int(target: Target) -> int:
        """尽量把 target_type 转成 int，兼容 int / float / Enum。"""
        target_type = target.target_type

        if hasattr(target_type, "value"):
            return int(target_type.value)

        return int(target_type)


def load_destroy_target_selection_config(
    config: dict[str, Any],
) -> DestroyTargetSelectionConfig:
    """从项目总配置中读取摧毁目标集选择配置。"""
    prefix = "destroy_target_selection"

    selector_config = DestroyTargetSelectionConfig(
        method=str(
            get_config_value(config, f"{prefix}.method", default="weighted_kmeans")
        ),
        num_clusters=int(
            get_config_value(config, f"{prefix}.num_clusters", default=2)
        ),
        max_iter=int(
            get_config_value(config, f"{prefix}.max_iter", default=300)
        ),
        tolerance=float(
            get_config_value(config, f"{prefix}.tolerance", default=0.0004)
        ),
        random_seed=int(
            get_config_value(config, f"{prefix}.random_seed", default=42)
        ),
        normalized_distance_weight=float(
            get_config_value(
                config,
                f"{prefix}.feature_weights.normalized_distance",
                default=0.4,
            )
        ),
        score_weight=float(
            get_config_value(
                config,
                f"{prefix}.feature_weights.score",
                default=0.6,
            )
        ),
        choose_cluster_by=str(
            get_config_value(
                config,
                f"{prefix}.choose_cluster_by",
                default="highest_score_cluster",
            )
        ),

        rf_model_path=str(
            get_config_value(
                config,
                f"{prefix}.random_forest.model_path",
                default="checkpoints/random_forest/destroy_target_rf.joblib",
            )
        ),
        rf_probability_threshold=float(
            get_config_value(
                config,
                f"{prefix}.random_forest.probability_threshold",
                default=0.5,
            )
        ),
        rf_fallback_method=str(
            get_config_value(
                config,
                f"{prefix}.random_forest.fallback_method",
                default="top_k",
            )
        ),
        rf_fallback_top_k=int(
            get_config_value(
                config,
                f"{prefix}.random_forest.fallback_top_k",
                default=5,
            )
        ),
        rf_feature_names=tuple(
            get_config_value(
                config,
                f"{prefix}.random_forest.feature_names",
                default=[
                    "score",
                    "normalized_distance",
                    "angle",
                    "adjusted_defense",
                    "adjusted_significance",
                ],
            )
        ),

        rule_top_k=int(
            get_config_value(config, f"{prefix}.rule.top_k", default=8)
        ),
        rule_threshold=float(
            get_config_value(config, f"{prefix}.rule.threshold", default=0.5)
        ),

        debug_csv_path=str(
            get_config_value(
                config,
                f"{prefix}.output.debug_csv_path",
                default="outputs/intermediate/destroy_target_selection.csv",
            )
        ),
    )

    selector_config.validate()
    return selector_config
