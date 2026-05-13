from copy import deepcopy

from uav_dynamic_task_allocation.allocation.target_clustering import (
    TargetClusterer,
    load_target_clustering_config,
)
from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.preprocessing.destroy_target_selection import (
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
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


def set_nested_config_value(config, key_path: str, value) -> None:
    """修改嵌套配置，用于临时切换 target_clustering.method。"""
    keys = key_path.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def run_clustering_check(config, method: str, logger, project_root):
    """运行指定目标分群方法。"""
    local_config = deepcopy(config)
    set_nested_config_value(local_config, "target_clustering.method", method)

    data = load_all_data(local_config)

    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    screening_config = load_target_screening_config(local_config)
    screener = TargetScreener(screening_config)
    screened_set = screener.screen(state.targets)

    selector_config = load_destroy_target_selection_config(local_config)
    selector = DestroyTargetSelector(selector_config)
    selection_result = selector.select(screened_set)
    selected_screened_set = selection_result.to_screened_target_set()

    clustering_config = load_target_clustering_config(local_config)
    clusterer = TargetClusterer(clustering_config)

    clustering_result = clusterer.cluster(selected_screened_set)

    csv_path = clusterer.write_debug_csv(
        result=clustering_result,
        project_root=project_root,
    )

    logger.info("=" * 80)
    logger.info(f"Target clustering method: {method}")
    logger.info(f"Destroy targets: {len(selected_screened_set.selected_targets)}")
    logger.info(f"Number of clusters: {clustering_result.cluster_set.num_clusters}")
    logger.info(f"Debug CSV saved to: {csv_path}")
    logger.info(f"Metadata: {clustering_result.metadata}")

    for cluster in clustering_result.cluster_set.clusters:
        logger.info(
            f"Cluster {cluster.cluster_id}: "
            f"num_targets={cluster.num_targets}, "
            f"target_ids={cluster.target_ids}, "
            f"center=({cluster.center.x:.4f}, {cluster.center.y:.4f}), "
            f"defense_sum={cluster.defense_sum:.4f}, "
            f"significance_sum={cluster.significance_sum:.4f}, "
            f"compactness={cluster.compactness:.4f}"
        )


def main() -> None:
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("Target clustering check started.")

    for method in ["kmeans", "pso"]:
        run_clustering_check(
            config=config,
            method=method,
            logger=logger,
            project_root=project_root,
        )

    logger.info("Target clustering check finished successfully.")


if __name__ == "__main__":
    main()