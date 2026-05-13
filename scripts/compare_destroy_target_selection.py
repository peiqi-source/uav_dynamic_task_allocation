"""comparedestroy目标selection脚本，封装可直接运行的实验、检查或可视化流程。"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.preprocessing.destroy_target_selection import (
    DestroyTargetSelectionError,
    DestroyTargetSelectionResult,
    DestroyTargetSelector,
    load_destroy_target_selection_config,
)
from uav_dynamic_task_allocation.preprocessing.target_screening import (
    TargetScreener,
    load_target_screening_config,
)
from uav_dynamic_task_allocation.utils.config import (
    get_project_root,
    load_and_validate_config,
    resolve_path,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    默认对比四种摧毁目标集决策方法：
    weighted_kmeans、random_forest、top_k、threshold。
    """
    parser = argparse.ArgumentParser(
        description="Compare destroy target selection algorithms."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file.",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["weighted_kmeans", "random_forest", "top_k", "threshold"],
        help=(
            "Methods to compare. "
            "Available: weighted_kmeans random_forest top_k threshold"
        ),
    )

    parser.add_argument(
        "--detail-output",
        type=str,
        default="outputs/evaluation/destroy_target_selection_comparison.csv",
        help="Output CSV path for target-level comparison.",
    )

    parser.add_argument(
        "--summary-output",
        type=str,
        default="outputs/evaluation/destroy_target_selection_summary.csv",
        help="Output CSV path for method-level summary.",
    )

    return parser.parse_args()


def resolve_config_path(config_path: str) -> Path:
    """解析配置文件路径。"""
    path = Path(config_path)

    if path.is_absolute():
        return path

    return get_project_root() / path


def set_nested_config_value(
    config: dict[str, Any],
    key_path: str,
    value: Any,
) -> None:
    """
    修改嵌套配置。

    这里用于临时切换 destroy_target_selection.method，
    不会写回 default.yaml。
    """
    keys = key_path.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def run_selection_method(
    base_config: dict[str, Any],
    screened_set,
    method: str,
) -> DestroyTargetSelectionResult:
    """
    使用指定方法运行摧毁目标集选择。

    每次都 deepcopy 一份 config，避免不同方法之间互相污染配置。
    """
    config = deepcopy(base_config)
    set_nested_config_value(
        config,
        "destroy_target_selection.method",
        method,
    )

    selector_config = load_destroy_target_selection_config(config)
    selector = DestroyTargetSelector(selector_config)

    return selector.select(screened_set)


def target_type_to_int(target) -> int:
    """将 target_type 尽量转成 int。"""
    if hasattr(target.target_type, "value"):
        return int(target.target_type.value)

    return int(target.target_type)


def build_detail_rows(
    screened_set,
    method_results: dict[str, DestroyTargetSelectionResult],
) -> list[dict[str, Any]]:
    """
    构造目标级对比表。

    每一行对应一个 target。
    每种算法会增加一列：
        selected_by_weighted_kmeans
        selected_by_random_forest
        selected_by_top_k
        selected_by_threshold
    """
    selected_ids_by_method = {
        method: {int(target.target_id) for target in result.destroy_targets}
        for method, result in method_results.items()
    }

    rows: list[dict[str, Any]] = []

    for target in screened_set.candidate_targets:
        target_id = int(target.target_id)
        target_score = screened_set.target_scores[target_id]
        components = target_score.components
        metadata = target_score.metadata

        row: dict[str, Any] = {
            "target_id": target_id,
            "target_type": target_type_to_int(target),
            "x": float(target.position.x),
            "y": float(target.position.y),
            "defense": float(target.defense),
            "significance": float(target.significance),
            "score": float(target_score.score),
            "raw_score": components.get("raw_score", ""),
            "normalized_distance": components.get("normalized_distance", ""),
            "angle": components.get("angle", ""),
            "adjusted_defense": components.get("adjusted_defense", ""),
            "adjusted_significance": components.get("adjusted_significance", ""),
            "affected_by_defense_posts": json.dumps(
                metadata.get("affected_by_defense_posts", []),
                ensure_ascii=False,
            ),
        }

        selected_count = 0

        for method, selected_ids in selected_ids_by_method.items():
            selected = target_id in selected_ids
            row[f"selected_by_{method}"] = int(selected)

            if selected:
                selected_count += 1

        row["selected_method_count"] = selected_count

        rows.append(row)

    rows.sort(
        key=lambda item: (
            item["selected_method_count"],
            item["score"],
        ),
        reverse=True,
    )

    return rows


def build_summary_rows(
    screened_set,
    method_results: dict[str, DestroyTargetSelectionResult],
) -> list[dict[str, Any]]:
    """
    构造算法级 summary。

    每一行对应一种算法，统计：
    - 选中目标数量；
    - 选中目标平均 score；
    - 平均 defense；
    - 平均 significance；
    - 平均 normalized_distance；
    - target ids。
    """
    rows: list[dict[str, Any]] = []

    total_candidate_targets = len(screened_set.candidate_targets)

    for method, result in method_results.items():
        destroy_targets = result.destroy_targets

        if destroy_targets:
            scores = [
                float(screened_set.target_scores[int(target.target_id)].score)
                for target in destroy_targets
            ]
            defenses = [
                float(target.defense)
                for target in destroy_targets
            ]
            significances = [
                float(target.significance)
                for target in destroy_targets
            ]
            normalized_distances = [
                float(
                    screened_set.target_scores[int(target.target_id)]
                    .components.get("normalized_distance", 0.0)
                )
                for target in destroy_targets
            ]

            avg_score = sum(scores) / len(scores)
            avg_defense = sum(defenses) / len(defenses)
            avg_significance = sum(significances) / len(significances)
            avg_normalized_distance = (
                sum(normalized_distances) / len(normalized_distances)
            )
        else:
            avg_score = 0.0
            avg_defense = 0.0
            avg_significance = 0.0
            avg_normalized_distance = 0.0

        destroy_ids = [
            int(target.target_id)
            for target in destroy_targets
        ]

        rows.append(
            {
                "method": method,
                "num_destroy_targets": len(destroy_targets),
                "num_non_destroy_targets": len(result.non_destroy_targets),
                "selection_ratio": (
                    len(destroy_targets) / total_candidate_targets
                    if total_candidate_targets > 0
                    else 0.0
                ),
                "avg_score": avg_score,
                "avg_defense": avg_defense,
                "avg_significance": avg_significance,
                "avg_normalized_distance": avg_normalized_distance,
                "destroy_target_ids": json.dumps(destroy_ids, ensure_ascii=False),
                "algorithm_type": result.algorithm_metadata.algorithm_type,
                "metadata": json.dumps(result.metadata, ensure_ascii=False),
            }
        )

    return rows


def build_overlap_rows(
    method_results: dict[str, DestroyTargetSelectionResult],
) -> list[dict[str, Any]]:
    """
    构造方法之间的目标集重合度统计。

    当前这个函数先返回 rows，后面如果需要可以单独保存为 CSV。
    """
    methods = list(method_results.keys())

    selected_ids_by_method = {
        method: {int(target.target_id) for target in result.destroy_targets}
        for method, result in method_results.items()
    }

    rows: list[dict[str, Any]] = []

    for method_a in methods:
        for method_b in methods:
            ids_a = selected_ids_by_method[method_a]
            ids_b = selected_ids_by_method[method_b]

            intersection = ids_a & ids_b
            union = ids_a | ids_b

            jaccard = len(intersection) / len(union) if union else 0.0

            rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                    "jaccard": jaccard,
                    "intersection_ids": json.dumps(
                        sorted(intersection),
                        ensure_ascii=False,
                    ),
                }
            )

    return rows


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """保存 rows 为 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    """在控制台输出 summary。"""
    print("\n========== Destroy Target Selection Comparison ==========")

    for row in summary_rows:
        print(
            f"{row['method']:16s} | "
            f"num={row['num_destroy_targets']:3d} | "
            f"ratio={row['selection_ratio']:.3f} | "
            f"avg_score={row['avg_score']:.4f} | "
            f"avg_defense={row['avg_defense']:.4f} | "
            f"avg_significance={row['avg_significance']:.4f} | "
            f"avg_norm_dist={row['avg_normalized_distance']:.4f} | "
            f"ids={row['destroy_target_ids']}"
        )

    print("=========================================================\n")


def print_overlap(overlap_rows: list[dict[str, Any]]) -> None:
    """在控制台输出方法之间的重合度。"""
    print("\n========== Method Overlap Jaccard ==========")

    for row in overlap_rows:
        if row["method_a"] == row["method_b"]:
            continue

        print(
            f"{row['method_a']:16s} vs {row['method_b']:16s} | "
            f"intersection={row['intersection_count']:3d} | "
            f"union={row['union_count']:3d} | "
            f"jaccard={row['jaccard']:.4f} | "
            f"common={row['intersection_ids']}"
        )

    print("============================================\n")


def main() -> None:
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        无返回值；通过状态变更、文件输出或日志记录体现执行结果。
    """
    args = parse_args()

    project_root = get_project_root()
    config_path = resolve_config_path(args.config)
    detail_output_path = resolve_path(
        args.detail_output,
        project_root=project_root,
    )
    summary_output_path = resolve_path(
        args.summary_output,
        project_root=project_root,
    )

    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("=" * 80)
    logger.info("Destroy target selection comparison started.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Methods: {args.methods}")
    logger.info("=" * 80)

    data = load_all_data(config)

    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    screening_config = load_target_screening_config(config)
    screener = TargetScreener(screening_config)

    screened_set = screener.screen(state.targets)
    screened_set.validate()

    method_results: dict[str, DestroyTargetSelectionResult] = {}

    for method in args.methods:
        try:
            result = run_selection_method(
                base_config=config,
                screened_set=screened_set,
                method=method,
            )
            method_results[method] = result

            logger.info(
                "Method finished: "
                f"method={method}, "
                f"num_destroy_targets={len(result.destroy_targets)}, "
                f"ids={[int(target.target_id) for target in result.destroy_targets]}"
            )

        except DestroyTargetSelectionError as exc:
            logger.error(
                f"Method failed: method={method}, error={exc}"
            )

    if not method_results:
        raise RuntimeError("No destroy target selection method succeeded.")

    detail_rows = build_detail_rows(
        screened_set=screened_set,
        method_results=method_results,
    )

    summary_rows = build_summary_rows(
        screened_set=screened_set,
        method_results=method_results,
    )

    overlap_rows = build_overlap_rows(
        method_results=method_results,
    )

    save_csv(detail_rows, detail_output_path)
    save_csv(summary_rows, summary_output_path)

    overlap_output_path = summary_output_path.with_name(
        summary_output_path.stem.replace("summary", "overlap")
        + summary_output_path.suffix
    )
    save_csv(overlap_rows, overlap_output_path)

    print_summary(summary_rows)
    print_overlap(overlap_rows)

    logger.info(f"Detail comparison saved to: {detail_output_path}")
    logger.info(f"Summary comparison saved to: {summary_output_path}")
    logger.info(f"Overlap comparison saved to: {overlap_output_path}")
    logger.info("Destroy target selection comparison finished successfully.")


if __name__ == "__main__":
    main()