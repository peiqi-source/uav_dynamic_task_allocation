"""资源分配模块中的montecarlo资源evaluator实现。"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from uav_dynamic_task_allocation.core.contracts import (
    AllocationPlan,
    ClusterAssignment,
)
from uav_dynamic_task_allocation.core.entities import Target, UAV
from uav_dynamic_task_allocation.utils.config import get_config_value, resolve_path


class MonteCarloResourceEvaluatorError(Exception):
    """Raised when Monte Carlo resource assessment cannot be completed."""


@dataclass(frozen=True)
class MonteCarloResourceEvaluatorConfig:
    """Configuration for Monte Carlo cluster resource sufficiency assessment."""

    # num_simulations: numsimulations。
    num_simulations: int = 300
    # completion_threshold: completionthreshold。
    completion_threshold: float = 0.8
    # uav_loss_probability: 无人机损失值probability。
    uav_loss_probability: float = 0.05
    # target_defense_noise: 目标防御能力noise。
    target_defense_noise: float = 0.1
    # hit_probability_noise: hitprobabilitynoise。
    hit_probability_noise: float = 0.05
    # distance_consumption_weight: distanceconsumption权重。
    distance_consumption_weight: float = 0.05
    # min_hit_probability: 最小值hitprobability。
    min_hit_probability: float = 0.05
    # max_hit_probability: 最大值hitprobability。
    max_hit_probability: float = 0.98
    # random_seed: 随机随机种子。
    random_seed: int = 42
    # output_csv_path: 输出CSV 数据路径。
    output_csv_path: str = "outputs/resource_assessment/monte_carlo_results.csv"
    # figure_path: 图表路径。
    figure_path: str = "outputs/resource_assessment/monte_carlo_completion_distribution.png"

    def validate(self) -> None:
        """校验当前对象或输入配置的合法性。

        参数：
            无显式业务参数。

        返回：
            无返回值；通过状态变更、文件输出或日志记录体现执行结果。
        """
        if self.num_simulations <= 0:
            raise MonteCarloResourceEvaluatorError("num_simulations must be positive.")
        if not 0.0 <= self.completion_threshold <= 1.0:
            raise MonteCarloResourceEvaluatorError(
                "completion_threshold must be in [0, 1]."
            )
        for name in (
            "uav_loss_probability",
            "target_defense_noise",
            "hit_probability_noise",
            "min_hit_probability",
            "max_hit_probability",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise MonteCarloResourceEvaluatorError(f"{name} must be non-negative.")
        if self.min_hit_probability > self.max_hit_probability:
            raise MonteCarloResourceEvaluatorError(
                "min_hit_probability must be <= max_hit_probability."
            )


@dataclass
class ResourceAssessmentResult:
    """Monte Carlo resource assessment result for one target cluster."""

    # cluster_id: 目标簇编号。
    cluster_id: int
    # completion_probability: completionprobability。
    completion_probability: float
    # mean_completion_rate: 均值completion率。
    mean_completion_rate: float
    # std_completion_rate: 标准差completion率。
    std_completion_rate: float
    # quantile_05: quantile05。
    quantile_05: float
    # quantile_50: quantile50。
    quantile_50: float
    # quantile_95: quantile95。
    quantile_95: float
    # threshold: threshold 数据。
    threshold: float
    # threshold_passed: thresholdpassed。
    threshold_passed: bool
    # resource_sufficiency: 资源sufficiency。
    resource_sufficiency: str
    # support_required: 支援required。
    support_required: bool
    # recommended_support_type: recommended支援类型。
    recommended_support_type: str
    # details: details 数据。
    details: dict[str, Any] = field(default_factory=dict)
    # samples: samples 数据。
    samples: list[float] = field(default_factory=list)
    # trial_records: trialrecords。
    trial_records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_samples: bool = False) -> dict[str, Any]:
        """将对象转换为字典，便于日志记录、序列化或调试输出。

        参数：
            include_samples: includesamples，类型为 bool。

        返回：
            dict[str, Any]，表示该函数计算或构建得到的结果。
        """
        payload = asdict(self)
        if not include_samples:
            payload.pop("samples", None)
            payload.pop("trial_records", None)
        return payload


class MonteCarloResourceEvaluator:
    """
    Evaluate whether assigned UAV resources are likely to complete cluster tasks.

    The evaluator performs repeated stochastic trials. In each trial it perturbs
    target defense, randomly removes some assigned UAVs, estimates hit
    probability from remaining firepower versus perturbed defense, and reports
    the value-weighted completion rate.
    """

    def __init__(self, config: MonteCarloResourceEvaluatorConfig) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            config: 配置对象，类型为 MonteCarloResourceEvaluatorConfig。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        config.validate()
        # config: 配置。
        self.config = config
        # _rng: rng 数据。
        self._rng = np.random.default_rng(config.random_seed)

    def assess_assignment(
        self,
        assignment: ClusterAssignment,
    ) -> ResourceAssessmentResult:
        """处理assessassignment相关业务逻辑。

        参数：
            assignment: assignment 数据，类型为 ClusterAssignment。

        返回：
            ResourceAssessmentResult，表示该函数计算或构建得到的结果。
        """
        assignment.validate()

        targets = list(assignment.target_cluster.targets)
        attack_uavs = list(assignment.assigned_attack_uavs)
        guide_uavs = list(assignment.assigned_guide_uavs)
        communication_uavs = list(assignment.assigned_communication_uavs)

        if not targets:
            raise MonteCarloResourceEvaluatorError(
                f"Cluster {assignment.cluster_id} has no targets."
            )

        trial_records = [
            self._simulate_once(
                targets=targets,
                attack_uavs=attack_uavs,
                assignment=assignment,
            )
            for _ in range(self.config.num_simulations)
        ]
        samples = [
            float(record["completion_rate"])
            for record in trial_records
        ]
        sample_array = np.asarray(samples, dtype=np.float64)

        completion_probability = float(
            np.mean(sample_array >= self.config.completion_threshold)
        )
        mean_completion_rate = float(mean(samples))
        std_completion_rate = float(pstdev(samples)) if len(samples) > 1 else 0.0
        quantiles = np.quantile(sample_array, [0.05, 0.5, 0.95])
        threshold_passed = completion_probability >= self.config.completion_threshold

        recommended_support_type = self._recommend_support_type(
            attack_uavs=attack_uavs,
            guide_uavs=guide_uavs,
            communication_uavs=communication_uavs,
            mean_completion_rate=mean_completion_rate,
            threshold_passed=threshold_passed,
        )
        expected_surviving_attack_uavs = float(
            mean(float(record["surviving_attack_uavs"]) for record in trial_records)
        )
        expected_destroyed_value = float(
            mean(float(record["destroyed_value"]) for record in trial_records)
        )
        resource_redundancy_ratio = self._resource_redundancy_ratio(assignment)
        support_trigger_score = self._support_trigger_score(
            completion_probability=completion_probability,
            mean_completion_rate=mean_completion_rate,
            resource_redundancy_ratio=resource_redundancy_ratio,
        )
        shortage_reasons = self._shortage_reasons(
            attack_uavs=attack_uavs,
            guide_uavs=guide_uavs,
            communication_uavs=communication_uavs,
            resource_redundancy_ratio=resource_redundancy_ratio,
            threshold_passed=threshold_passed,
        )

        return ResourceAssessmentResult(
            cluster_id=assignment.cluster_id,
            completion_probability=completion_probability,
            mean_completion_rate=mean_completion_rate,
            std_completion_rate=std_completion_rate,
            quantile_05=float(quantiles[0]),
            quantile_50=float(quantiles[1]),
            quantile_95=float(quantiles[2]),
            threshold=self.config.completion_threshold,
            threshold_passed=threshold_passed,
            resource_sufficiency=(
                "sufficient" if threshold_passed else "insufficient"
            ),
            support_required=not threshold_passed,
            recommended_support_type=recommended_support_type,
            details={
                "num_targets": len(targets),
                "num_attack_uavs": len(attack_uavs),
                "num_guide_uavs": len(guide_uavs),
                "num_communication_uavs": len(communication_uavs),
                "target_ids": [target.target_id for target in targets],
                "assigned_attack_uav_ids": [
                    uav.uav_id for uav in attack_uavs
                ],
                "required_attack_uav_count": assignment.required_attack_uav_count,
                "resource_redundancy_ratio": resource_redundancy_ratio,
                "expected_surviving_attack_uavs": expected_surviving_attack_uavs,
                "expected_destroyed_value": expected_destroyed_value,
                "support_trigger_score": support_trigger_score,
                "shortage_reasons": shortage_reasons,
            },
            samples=[float(value) for value in samples],
            trial_records=trial_records,
        )

    def assess_allocation(
        self,
        allocation_plan: AllocationPlan,
    ) -> list[ResourceAssessmentResult]:
        """处理assess资源分配相关业务逻辑。

        参数：
            allocation_plan: 资源分配规划方案，类型为 AllocationPlan。

        返回：
            list[ResourceAssessmentResult]，表示该函数计算或构建得到的结果。
        """
        allocation_plan.validate()
        return [
            self.assess_assignment(assignment)
            for assignment in allocation_plan.assignments
        ]

    def write_csv(
        self,
        results: list[ResourceAssessmentResult],
        output_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> Path:
        """处理writeCSV 数据相关业务逻辑。

        参数：
            results: 结果集合，类型为 list[ResourceAssessmentResult]。
            output_path: 输出路径，类型为 str | Path | None。
            project_root: projectroot，类型为 str | Path | None。

        返回：
            Path，表示该函数计算或构建得到的结果。
        """
        path = resolve_path(
            output_path or self.config.output_csv_path,
            project_root=project_root,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "cluster_id",
            "completion_probability",
            "mean_completion_rate",
            "std_completion_rate",
            "quantile_05",
            "quantile_50",
            "quantile_95",
            "threshold",
            "threshold_passed",
            "resource_sufficiency",
            "support_required",
            "recommended_support_type",
            "details",
        ]

        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                row = result.to_dict()
                writer.writerow(row)

        return path

    def write_samples_csv(
        self,
        results: list[ResourceAssessmentResult],
        output_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> Path:
        """Write per-trial Monte Carlo samples for audit and plotting."""
        default_path = Path(self.config.output_csv_path)
        sample_path = default_path.with_name(f"{default_path.stem}_samples.csv")
        path = resolve_path(output_path or sample_path, project_root=project_root)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "cluster_id",
            "trial_index",
            "completion_rate",
            "destroyed_value",
            "total_value",
            "surviving_attack_uavs",
            "mean_hit_probability",
            "distance_consumption",
        ]
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                for trial_index, record in enumerate(result.trial_records):
                    writer.writerow(
                        {
                            "cluster_id": result.cluster_id,
                            "trial_index": trial_index,
                            **record,
                        }
                    )
        return path

    def write_distribution_plot(
        self,
        results: list[ResourceAssessmentResult],
        output_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> Path | None:
        """处理writedistribution绘图相关业务逻辑。

        参数：
            results: 结果集合，类型为 list[ResourceAssessmentResult]。
            output_path: 输出路径，类型为 str | Path | None。
            project_root: projectroot，类型为 str | Path | None。

        返回：
            Path | None，表示该函数计算或构建得到的结果。
        """
        if not results:
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        path = resolve_path(
            output_path or self.config.figure_path,
            project_root=project_root,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        for result in results:
            plt.hist(
                result.samples,
                bins=20,
                alpha=0.45,
                label=f"cluster {result.cluster_id}",
            )
        plt.axvline(
            self.config.completion_threshold,
            color="red",
            linestyle="--",
            label="threshold",
        )
        plt.xlabel("Completion rate")
        plt.ylabel("Frequency")
        plt.title("Monte Carlo resource assessment")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def _simulate_once(
        self,
        targets: list[Target],
        attack_uavs: list[UAV],
        assignment: ClusterAssignment,
    ) -> dict[str, float]:
        """处理simulateonce相关业务逻辑。

        参数：
            targets: 目标集合，类型为 list[Target]。
            attack_uavs: 攻击uavs，类型为 list[UAV]。
            assignment: assignment 数据，类型为 ClusterAssignment。

        返回：
            dict[str, float]，表示该函数计算或构建得到的结果。
        """
        available_uavs = [
            uav for uav in attack_uavs
            if self._rng.random() >= self.config.uav_loss_probability
        ]
        total_value = sum(max(float(target.significance), 1e-6) for target in targets)
        if not available_uavs or total_value <= 0:
            return {
                "completion_rate": 0.0,
                "destroyed_value": 0.0,
                "total_value": float(total_value),
                "surviving_attack_uavs": float(len(available_uavs)),
                "mean_hit_probability": 0.0,
                "distance_consumption": 0.0,
            }

        destroyed_value = 0.0
        firepower = sum(max(float(uav.attack_power), 0.0) for uav in available_uavs)
        per_target_firepower = firepower / max(len(targets), 1)
        hit_probabilities: list[float] = []
        distance_consumption = self._distance_consumption(
            assignment=assignment,
            attack_uavs=available_uavs,
        )
        effective_firepower = per_target_firepower * max(
            0.0,
            1.0 - self.config.distance_consumption_weight * distance_consumption,
        )

        for target in targets:
            defense_multiplier = max(
                0.0,
                float(
                    self._rng.normal(
                        loc=1.0,
                        scale=self.config.target_defense_noise,
                    )
                ),
            )
            perturbed_defense = max(float(target.defense) * defense_multiplier, 1e-6)
            base_hit_probability = effective_firepower / (
                effective_firepower + perturbed_defense
            )
            noisy_probability = float(
                self._rng.normal(
                    loc=base_hit_probability,
                    scale=self.config.hit_probability_noise,
                )
            )
            hit_probability = float(
                np.clip(
                    noisy_probability,
                    self.config.min_hit_probability,
                    self.config.max_hit_probability,
                )
            )
            if self._rng.random() <= hit_probability:
                destroyed_value += max(float(target.significance), 1e-6)
            hit_probabilities.append(hit_probability)

        return {
            "completion_rate": float(np.clip(destroyed_value / total_value, 0.0, 1.0)),
            "destroyed_value": float(destroyed_value),
            "total_value": float(total_value),
            "surviving_attack_uavs": float(len(available_uavs)),
            "mean_hit_probability": float(mean(hit_probabilities)),
            "distance_consumption": float(distance_consumption),
        }

    @staticmethod
    def _resource_redundancy_ratio(assignment: ClusterAssignment) -> float:
        """处理资源redundancyratio相关业务逻辑。

        参数：
            assignment: assignment 数据，类型为 ClusterAssignment。

        返回：
            float，表示该函数计算或构建得到的结果。
        """
        required = max(int(assignment.required_attack_uav_count), 1)
        assigned = len(assignment.assigned_attack_uavs)
        return float((assigned - required) / required)

    @staticmethod
    def _distance_consumption(
        assignment: ClusterAssignment,
        attack_uavs: list[UAV],
    ) -> float:
        """处理distanceconsumption相关业务逻辑。

        参数：
            assignment: assignment 数据，类型为 ClusterAssignment。
            attack_uavs: 攻击uavs，类型为 list[UAV]。

        返回：
            float，表示该函数计算或构建得到的结果。
        """
        center = assignment.target_cluster.center
        if not attack_uavs:
            return 0.0
        ratios = []
        for uav in attack_uavs:
            if uav.work_range <= 0:
                continue
            ratios.append(uav.position.distance_to(center) / float(uav.work_range))
        if not ratios:
            return 0.0
        return float(np.clip(mean(ratios), 0.0, 1.0))

    def _support_trigger_score(
        self,
        completion_probability: float,
        mean_completion_rate: float,
        resource_redundancy_ratio: float,
    ) -> float:
        """处理支援trigger评分相关业务逻辑。

        参数：
            completion_probability: completionprobability，类型为 float。
            mean_completion_rate: 均值completion率，类型为 float。
            resource_redundancy_ratio: 资源redundancyratio，类型为 float。

        返回：
            float，表示该函数计算或构建得到的结果。
        """
        probability_gap = max(0.0, self.config.completion_threshold - completion_probability)
        mean_gap = max(0.0, self.config.completion_threshold - mean_completion_rate)
        shortage_gap = max(0.0, -resource_redundancy_ratio)
        return float(np.clip(0.5 * probability_gap + 0.3 * mean_gap + 0.2 * shortage_gap, 0.0, 1.0))

    @staticmethod
    def _shortage_reasons(
        attack_uavs: list[UAV],
        guide_uavs: list[UAV],
        communication_uavs: list[UAV],
        resource_redundancy_ratio: float,
        threshold_passed: bool,
    ) -> list[str]:
        """处理shortagereasons相关业务逻辑。

        参数：
            attack_uavs: 攻击uavs，类型为 list[UAV]。
            guide_uavs: 导引uavs，类型为 list[UAV]。
            communication_uavs: 通信uavs，类型为 list[UAV]。
            resource_redundancy_ratio: 资源redundancyratio，类型为 float。
            threshold_passed: thresholdpassed，类型为 bool。

        返回：
            list[str]，表示该函数计算或构建得到的结果。
        """
        reasons: list[str] = []
        if not threshold_passed:
            reasons.append("completion_probability_below_threshold")
        if resource_redundancy_ratio < 0:
            reasons.append("attack_resource_shortage")
        if not attack_uavs:
            reasons.append("no_attack_uav")
        if not guide_uavs:
            reasons.append("no_recon_uav")
        if not communication_uavs:
            reasons.append("no_communication_uav")
        return reasons

    @staticmethod
    def _recommend_support_type(
        attack_uavs: list[UAV],
        guide_uavs: list[UAV],
        communication_uavs: list[UAV],
        mean_completion_rate: float,
        threshold_passed: bool,
    ) -> str:
        """处理recommend支援类型相关业务逻辑。

        参数：
            attack_uavs: 攻击uavs，类型为 list[UAV]。
            guide_uavs: 导引uavs，类型为 list[UAV]。
            communication_uavs: 通信uavs，类型为 list[UAV]。
            mean_completion_rate: 均值completion率，类型为 float。
            threshold_passed: thresholdpassed，类型为 bool。

        返回：
            str，表示该函数计算或构建得到的结果。
        """
        if threshold_passed:
            return "no_support"
        if not attack_uavs or mean_completion_rate < 0.5:
            return "fire_support"
        if not guide_uavs:
            return "recon_support"
        if not communication_uavs:
            return "communication_support"
        return "task_reassignment"


def load_monte_carlo_resource_evaluator_config(
    config: dict[str, Any],
) -> MonteCarloResourceEvaluatorConfig:
    """Load Monte Carlo resource evaluator configuration from project config."""
    prefix = "monte_carlo"
    compat_prefix = "resource_allocation.monte_carlo"

    evaluator_config = MonteCarloResourceEvaluatorConfig(
        num_simulations=int(
            get_config_value(
                config,
                f"{prefix}.num_simulations",
                default=get_config_value(
                    config,
                    f"{compat_prefix}.num_simulations",
                    default=300,
                ),
            )
        ),
        completion_threshold=float(
            get_config_value(
                config,
                f"{prefix}.completion_threshold",
                default=get_config_value(
                    config,
                    f"{compat_prefix}.completion_threshold",
                    default=0.8,
                ),
            )
        ),
        uav_loss_probability=float(
            get_config_value(
                config,
                f"{prefix}.uav_loss_probability",
                default=get_config_value(
                    config,
                    f"{compat_prefix}.uav_loss_probability",
                    default=0.05,
                ),
            )
        ),
        target_defense_noise=float(
            get_config_value(
                config,
                f"{prefix}.target_defense_noise",
                default=get_config_value(
                    config,
                    f"{compat_prefix}.target_defense_noise",
                    default=0.1,
                ),
            )
        ),
        hit_probability_noise=float(
            get_config_value(
                config,
                f"{prefix}.hit_probability_noise",
                default=get_config_value(
                    config,
                    f"{compat_prefix}.hit_probability_noise",
                    default=0.05,
                ),
            )
        ),
        distance_consumption_weight=float(
            get_config_value(
                config,
                f"{prefix}.distance_consumption_weight",
                default=0.05,
            )
        ),
        random_seed=int(
            get_config_value(
                config,
                f"{prefix}.random_seed",
                default=get_config_value(config, "project.seed", default=42),
            )
        ),
        output_csv_path=str(
            get_config_value(
                config,
                f"{prefix}.output_csv_path",
                default="outputs/resource_assessment/monte_carlo_results.csv",
            )
        ),
        figure_path=str(
            get_config_value(
                config,
                f"{prefix}.figure_path",
                default=(
                    "outputs/resource_assessment/"
                    "monte_carlo_completion_distribution.png"
                ),
            )
        ),
    )
    evaluator_config.validate()
    return evaluator_config
