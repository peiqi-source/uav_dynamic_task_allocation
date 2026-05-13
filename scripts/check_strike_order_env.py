from uav_dynamic_task_allocation.core.contracts import TargetCluster
from uav_dynamic_task_allocation.core.entities import Position, build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.envs.strike_order_env import (
    StrikeOrderEnv,
    load_strike_order_env_config,
)
from uav_dynamic_task_allocation.utils.config import (
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


def build_debug_cluster(targets, cluster_id: int = 0, max_targets: int = 6) -> TargetCluster:
    """
    构造一个用于检查 StrikeOrderEnv 的目标群。

    当前只是从真实数据中取前 max_targets 个目标。
    后续真正流程会由 target_clustering.py 生成 TargetCluster。
    """
    selected_targets = targets[:max_targets]

    if not selected_targets:
        raise RuntimeError("No targets available to build debug cluster.")

    center = Position(
        x=sum(target.position.x for target in selected_targets) / len(selected_targets),
        y=sum(target.position.y for target in selected_targets) / len(selected_targets),
    )

    cluster = TargetCluster(
        cluster_id=cluster_id,
        targets=selected_targets,
        center=center,
        defense_sum=sum(target.defense for target in selected_targets),
        significance_sum=sum(target.significance for target in selected_targets),
        compactness=None,
        metadata={"source": "check_strike_order_env.py"},
    )

    cluster.validate()
    return cluster


def select_nearest_valid_action(env: StrikeOrderEnv) -> int:
    """
    选择距离当前位置最近的未访问目标。

    这不是 DQN，只是用于检查环境 step 是否正常。
    """
    mask = env.get_action_mask()
    valid_action_ids = [index for index, value in enumerate(mask) if value > 0]

    if not valid_action_ids:
        raise RuntimeError("No valid action available.")

    best_action_id = valid_action_ids[0]
    best_distance = float("inf")

    for action_id in valid_action_ids:
        target = env.target_cluster.targets[action_id]
        distance = env._distance(env.current_position, target.position)

        if distance < best_distance:
            best_distance = distance
            best_action_id = action_id

    return best_action_id


def main() -> None:
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("StrikeOrderEnv check started.")

    data = load_all_data(config)
    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    strike_order_config = load_strike_order_env_config(config)

    cluster = build_debug_cluster(
        targets=state.targets,
        cluster_id=0,
        max_targets=6,
    )

    # DQN 训练时可以认为 UAV 小组已经到达目标群入口。
    # 这里用目标群中心作为调试起点，也可以改成基地或某个分配小组当前位置。
    start_position = cluster.center

    env = StrikeOrderEnv(
        target_cluster=cluster,
        start_position=start_position,
        config=strike_order_config,
    )

    observation = env.reset()

    logger.info("Environment reset successfully.")
    logger.info(f"Cluster id: {cluster.cluster_id}")
    logger.info(f"Target ids: {cluster.target_ids}")
    logger.info(f"Start position: ({start_position.x}, {start_position.y})")
    logger.info(f"Observation shape: {observation.vector.shape}")
    logger.info(f"Action mask shape: {observation.action_mask.shape}")
    logger.info(f"Initial debug info: {observation.debug_info}")
    logger.info(f"Initial action mask: {observation.action_mask.tolist()}")

    while True:
        action_id = select_nearest_valid_action(env)
        result = env.step(action_id)

        logger.info(
            "Step result: "
            f"action_id={action_id}, "
            f"reward={result.reward:.6f}, "
            f"done={result.done}, "
            f"info={result.info}"
        )

        if result.done:
            break

    plan_dict = env.get_strike_order_plan_dict()

    logger.info("Strike order environment episode finished.")
    logger.info(f"Generated strike order: {plan_dict}")
    logger.info("StrikeOrderEnv check finished successfully.")


if __name__ == "__main__":
    main()