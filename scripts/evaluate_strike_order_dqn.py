"""评估打击顺序DQN 算法脚本，封装可直接运行的实验、检查或可视化流程。"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch

    from uav_dynamic_task_allocation.algorithms.rl.dqn.agent import (
        DQNAgent,
        DQNAgentConfig,
    )
    from uav_dynamic_task_allocation.algorithms.rl.dqn.checkpoint import (
        DQNCheckpointConfig,
        DQNCheckpointManager,
    )
    from uav_dynamic_task_allocation.algorithms.rl.dqn.network import (
        DQNNetworkConfig,
    )
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False
from uav_dynamic_task_allocation.core.contracts import TargetCluster
from uav_dynamic_task_allocation.core.entities import Position, build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.envs.strike_order_env import (
    StrikeOrderEnv,
    load_strike_order_env_config,
)
from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.device import get_device
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config
from uav_dynamic_task_allocation.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Evaluate trained StrikeOrder DQN against baseline methods."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        choices=["best", "latest"],
        help="Which checkpoint to evaluate.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override experiment.device.",
    )

    parser.add_argument(
        "--random-trials",
        type=int,
        default=50,
        help="Number of random baseline trials per cluster.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/evaluation/strike_order_dqn_evaluation.csv",
        help="Output CSV path.",
    )

    return parser.parse_args()


def set_nested_config_value(
    config: dict[str, Any],
    key_path: str,
    value: Any,
) -> None:
    """修改嵌套配置字典中的某个值。"""
    keys = key_path.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def resolve_path(path: str) -> Path:
    """
    解析路径。

    如果是相对路径，则默认相对于项目根目录。
    """
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    return get_project_root() / path_obj


def apply_command_line_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """根据命令行参数临时覆盖配置。"""
    if args.device is not None:
        set_nested_config_value(
            config,
            "experiment.device",
            args.device,
        )


def build_debug_clusters(
    targets,
    num_clusters: int,
    targets_per_cluster: int,
) -> list[TargetCluster]:
    """
    用真实 Target 数据临时构造 debug clusters。

    注意：
    这必须尽量和 train_strike_order_dqn.py 中的 debug cluster 构造逻辑一致。
    后面 target_clustering.py 完成后，这里会替换为正式分群输出。
    """
    if num_clusters <= 0:
        raise RuntimeError("num_clusters must be positive.")

    if targets_per_cluster <= 0:
        raise RuntimeError("targets_per_cluster must be positive.")

    if not targets:
        raise RuntimeError("No targets available to build debug clusters.")

    total_targets = len(targets)
    requested_targets = num_clusters * targets_per_cluster
    used_targets = min(requested_targets, total_targets)

    selected_targets = targets[:used_targets]

    # 如果目标数量不足，自动调整为尽量均匀的 cluster。
    actual_num_clusters = min(num_clusters, used_targets)

    if actual_num_clusters <= 0:
        raise RuntimeError("actual_num_clusters must be positive.")

    target_indices = np.array_split(
        np.arange(len(selected_targets)),
        actual_num_clusters,
    )

    clusters: list[TargetCluster] = []

    for cluster_id, indices in enumerate(target_indices):
        cluster_targets = [selected_targets[int(index)] for index in indices]

        if not cluster_targets:
            continue

        center = Position(
            x=sum(target.position.x for target in cluster_targets)
            / len(cluster_targets),
            y=sum(target.position.y for target in cluster_targets)
            / len(cluster_targets),
        )

        cluster = TargetCluster(
            cluster_id=cluster_id,
            targets=cluster_targets,
            center=center,
            defense_sum=sum(target.defense for target in cluster_targets),
            significance_sum=sum(target.significance for target in cluster_targets),
            compactness=None,
            metadata={
                "source": "debug_cluster_builder",
                "requested_num_clusters": num_clusters,
                "requested_targets_per_cluster": targets_per_cluster,
                "actual_num_clusters": actual_num_clusters,
                "actual_num_targets": len(cluster_targets),
                "total_available_targets": total_targets,
            },
        )
        cluster.validate()
        clusters.append(cluster)

    if not clusters:
        raise RuntimeError("No debug clusters were created.")

    return clusters


def load_strike_order_network_config(
    config: dict[str, Any],
    input_dim: int,
    action_dim: int,
) -> DQNNetworkConfig:
    """读取 StrikeOrder DQN 网络配置。"""
    prefix = "strike_order_dqn.network"

    hidden_dims_raw = get_config_value(
        config,
        f"{prefix}.hidden_dims",
        default=[256, 256],
    )

    network_config = DQNNetworkConfig(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dims=tuple(int(value) for value in hidden_dims_raw),
        activation=str(
            get_config_value(config, f"{prefix}.activation", default="relu")
        ),
        dropout=float(
            get_config_value(config, f"{prefix}.dropout", default=0.0)
        ),
        use_layer_norm=bool(
            get_config_value(config, f"{prefix}.use_layer_norm", default=False)
        ),
        network_type=str(
            get_config_value(config, f"{prefix}.network_type", default="dqn")
        ),
    )

    network_config.validate()
    return network_config


def load_strike_order_agent_config(
    config: dict[str, Any],
) -> DQNAgentConfig:
    """读取 StrikeOrder DQN agent 配置。"""
    prefix = "strike_order_dqn.agent"

    agent_config = DQNAgentConfig(
        gamma=float(get_config_value(config, f"{prefix}.gamma", default=0.99)),
        learning_rate=float(
            get_config_value(config, f"{prefix}.learning_rate", default=5e-4)
        ),
        batch_size=int(
            get_config_value(config, f"{prefix}.batch_size", default=32)
        ),
        target_update_interval=int(
            get_config_value(
                config,
                f"{prefix}.target_update_interval",
                default=100,
            )
        ),
        epsilon_start=float(
            get_config_value(config, f"{prefix}.epsilon_start", default=1.0)
        ),
        epsilon_end=float(
            get_config_value(config, f"{prefix}.epsilon_end", default=0.05)
        ),
        epsilon_decay_steps=int(
            get_config_value(
                config,
                f"{prefix}.epsilon_decay_steps",
                default=3000,
            )
        ),
        gradient_clip_norm=float(
            get_config_value(
                config,
                f"{prefix}.gradient_clip_norm",
                default=10.0,
            )
        ),
        optimizer=str(
            get_config_value(config, f"{prefix}.optimizer", default="adam")
        ),
    )

    agent_config.validate()
    return agent_config


def load_strike_order_checkpoint_config(
    config: dict[str, Any],
) -> DQNCheckpointConfig:
    """读取 StrikeOrder DQN checkpoint 配置。"""
    prefix = "strike_order_dqn.checkpoint"

    checkpoint_config = DQNCheckpointConfig(
        enabled=bool(
            get_config_value(config, f"{prefix}.enabled", default=True)
        ),
        checkpoint_dir=str(
            get_config_value(
                config,
                f"{prefix}.checkpoint_dir",
                default="checkpoints/strike_order_dqn",
            )
        ),
        latest_filename=str(
            get_config_value(
                config,
                f"{prefix}.latest_filename",
                default="latest.pt",
            )
        ),
        best_filename=str(
            get_config_value(
                config,
                f"{prefix}.best_filename",
                default="best.pt",
            )
        ),
        save_optimizer=bool(
            get_config_value(
                config,
                f"{prefix}.save_optimizer",
                default=True,
            )
        ),
        save_replay_buffer=bool(
            get_config_value(
                config,
                f"{prefix}.save_replay_buffer",
                default=False,
            )
        ),
    )

    checkpoint_config.validate()
    return checkpoint_config


def select_nearest_action(env: StrikeOrderEnv) -> int:
    """
    最近邻 baseline。

    每一步选择离当前位置最近的未访问目标。
    """
    mask = env.get_action_mask()
    valid_action_ids = [index for index, value in enumerate(mask) if value > 0]

    if not valid_action_ids:
        raise RuntimeError("No valid action available for nearest baseline.")

    best_action_id = valid_action_ids[0]
    best_distance = float("inf")

    for action_id in valid_action_ids:
        target = env.target_cluster.targets[action_id]
        distance = env._distance(env.current_position, target.position)

        if distance < best_distance:
            best_distance = distance
            best_action_id = action_id

    return best_action_id


def select_random_action(
    env: StrikeOrderEnv,
    rng: np.random.Generator,
) -> int:
    """
    随机 baseline。

    每一步从当前未访问目标中随机选择一个。
    """
    mask = env.get_action_mask()
    valid_action_ids = np.where(mask > 0)[0]

    if valid_action_ids.size == 0:
        raise RuntimeError("No valid action available for random baseline.")

    return int(rng.choice(valid_action_ids))


def run_policy_episode(
    env: StrikeOrderEnv,
    method_name: str,
    agent: DQNAgent | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    在单个 TargetCluster 上运行一个策略。

    method_name 支持：
        dqn_greedy
        nearest_neighbor
        random
    """
    observation = env.reset()

    total_reward = 0.0
    num_steps = 0

    while True:
        if int(observation.action_mask.sum()) == 0:
            break

        if method_name == "dqn_greedy":
            if agent is None:
                raise RuntimeError("agent must be provided for dqn_greedy.")

            selection = agent.select_action(
                observation=observation.vector,
                action_mask=observation.action_mask,
                training=False,
            )
            action_id = selection.action_id

        elif method_name == "nearest_neighbor":
            action_id = select_nearest_action(env)

        elif method_name == "random":
            if rng is None:
                raise RuntimeError("rng must be provided for random policy.")

            action_id = select_random_action(env, rng)

        else:
            raise RuntimeError(f"Unsupported method_name: {method_name}")

        result = env.step(action_id)

        total_reward += result.reward
        num_steps += 1
        observation = result.observation

        if result.done:
            break

    plan_dict = env.get_strike_order_plan_dict()

    return {
        "method": method_name,
        "cluster_id": env.target_cluster.cluster_id,
        "num_targets": env.num_real_targets,
        "total_reward": total_reward,
        "total_path_distance": plan_dict["total_path_distance"],
        "num_steps": num_steps,
        "ordered_target_ids": plan_dict["ordered_target_ids"],
    }

def run_fixed_order_episode(
    cluster: TargetCluster,
    env_config,
    ordered_action_ids: list[int],
) -> dict[str, Any]:
    """
    按给定 action_id 顺序执行一个 StrikeOrder episode。

    这个函数用于：
    1. brute-force 枚举所有可能顺序；
    2. 评估某个固定目标顺序的 reward 和 path distance。

    注意：
    ordered_action_ids 是 cluster 内部的 target slot，不是 target_id。
    """
    env = StrikeOrderEnv(
        target_cluster=cluster,
        start_position=cluster.center,
        config=env_config,
    )

    observation = env.reset()

    total_reward = 0.0
    num_steps = 0

    for action_id in ordered_action_ids:
        if int(observation.action_mask.sum()) == 0:
            break

        result = env.step(action_id)

        total_reward += result.reward
        num_steps += 1
        observation = result.observation

        if result.done:
            break

    plan_dict = env.get_strike_order_plan_dict()

    return {
        "cluster_id": cluster.cluster_id,
        "num_targets": len(cluster.targets),
        "total_reward": float(total_reward),
        "total_path_distance": float(plan_dict["total_path_distance"]),
        "num_steps": num_steps,
        "ordered_target_ids": plan_dict["ordered_target_ids"],
        "ordered_action_ids": list(ordered_action_ids),
    }


def brute_force_optimal_orders(
    cluster: TargetCluster,
    env_config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    枚举当前 cluster 的所有目标访问顺序，找到两个最优 baseline。

    返回：
    1. reward_optimal:
       total_reward 最大的顺序。

    2. distance_optimal:
       total_path_distance 最短的顺序。

    对于当前 debug cluster，目标数量通常是 4 或 5，所以可以直接枚举。
    但如果以后一个 cluster 内目标很多，排列数量会爆炸，需要改成启发式或搜索算法。
    """
    num_targets = len(cluster.targets)

    if num_targets <= 0:
        raise RuntimeError("Cluster has no targets.")

    if num_targets > 8:
        raise RuntimeError(
            "Brute-force optimal order is too expensive for "
            f"num_targets={num_targets}. "
            "Please use this only for small debug clusters."
        )

    all_results: list[dict[str, Any]] = []

    for permutation in itertools.permutations(range(num_targets)):
        result = run_fixed_order_episode(
            cluster=cluster,
            env_config=env_config,
            ordered_action_ids=list(permutation),
        )
        all_results.append(result)

    reward_optimal = max(
        all_results,
        key=lambda row: row["total_reward"],
    )

    distance_optimal = min(
        all_results,
        key=lambda row: row["total_path_distance"],
    )

    reward_optimal = dict(reward_optimal)
    reward_optimal["method"] = "brute_force_reward_optimal"
    reward_optimal["num_permutations"] = len(all_results)

    distance_optimal = dict(distance_optimal)
    distance_optimal["method"] = "brute_force_distance_optimal"
    distance_optimal["num_permutations"] = len(all_results)

    return reward_optimal, distance_optimal


def evaluate_cluster(
    cluster: TargetCluster,
    env_config,
    agent: DQNAgent | None,
    rng: np.random.Generator,
    random_trials: int,
) -> list[dict[str, Any]]:
    """
    对一个 cluster 评估 DQN、nearest、random 和 brute-force optimal baseline。
    """
    rows: list[dict[str, Any]] = []

    start_position = cluster.center

    # 1. DQN greedy, when a torch checkpoint is available.
    if agent is not None:
        dqn_env = StrikeOrderEnv(
            target_cluster=cluster,
            start_position=start_position,
            config=env_config,
        )

        dqn_result = run_policy_episode(
            env=dqn_env,
            method_name="dqn_greedy",
            agent=agent,
        )
        rows.append(dqn_result)

    # 2. nearest-neighbor baseline
    nearest_env = StrikeOrderEnv(
        target_cluster=cluster,
        start_position=start_position,
        config=env_config,
    )

    nearest_result = run_policy_episode(
        env=nearest_env,
        method_name="nearest_neighbor",
    )
    rows.append(nearest_result)

    # 3. random baseline：保存平均值、best reward、best distance 对应的真实路径
    random_results: list[dict[str, Any]] = []

    for _ in range(random_trials):
        random_env = StrikeOrderEnv(
            target_cluster=cluster,
            start_position=start_position,
            config=env_config,
        )

        random_result = run_policy_episode(
            env=random_env,
            method_name="random",
            rng=rng,
        )

        random_results.append(random_result)

    random_rewards = [
        float(row["total_reward"])
        for row in random_results
    ]
    random_distances = [
        float(row["total_path_distance"])
        for row in random_results
    ]

    rows.append(
        {
            "method": "random_mean",
            "cluster_id": cluster.cluster_id,
            "num_targets": len(cluster.targets),
            "total_reward": float(np.mean(random_rewards)),
            "total_path_distance": float(np.mean(random_distances)),
            "num_steps": len(cluster.targets),
            "ordered_target_ids": [],
            "random_reward_std": float(np.std(random_rewards)),
            "random_distance_std": float(np.std(random_distances)),
            "random_trials": random_trials,
            "random_orders_sample": [
                row["ordered_target_ids"]
                for row in random_results[:3]
            ],
        }
    )

    random_best_reward = max(
        random_results,
        key=lambda row: float(row["total_reward"]),
    )
    random_best_reward = dict(random_best_reward)
    random_best_reward["method"] = "random_best_reward"
    random_best_reward["random_trials"] = random_trials

    random_best_distance = min(
        random_results,
        key=lambda row: float(row["total_path_distance"]),
    )
    random_best_distance = dict(random_best_distance)
    random_best_distance["method"] = "random_best_distance"
    random_best_distance["random_trials"] = random_trials

    rows.append(random_best_reward)
    rows.append(random_best_distance)

    # 4. brute-force optimal baseline
    reward_optimal, distance_optimal = brute_force_optimal_orders(
        cluster=cluster,
        env_config=env_config,
    )

    rows.append(reward_optimal)
    rows.append(distance_optimal)

    return rows


def save_evaluation_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """保存评估结果为 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "cluster_id",
        "method",
        "num_targets",
        "total_reward",
        "total_path_distance",
        "num_steps",
        "ordered_target_ids",
        "ordered_action_ids",
        "random_reward_std",
        "random_distance_std",
        "random_trials",
        "random_orders_sample",
        "num_permutations",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            csv_row = {}

            for field in fieldnames:
                value = row.get(field, "")

                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)

                csv_row[field] = value

            writer.writerow(csv_row)


def print_evaluation_summary(rows: list[dict[str, Any]]) -> None:
    """在控制台输出评估摘要。"""
    print("\n========== StrikeOrder DQN Evaluation Summary ==========")

    cluster_ids = sorted({int(row["cluster_id"]) for row in rows})

    for cluster_id in cluster_ids:
        cluster_rows = [
            row for row in rows
            if int(row["cluster_id"]) == cluster_id
        ]

        print(f"\nCluster {cluster_id}")

        for row in cluster_rows:
            method = row["method"]
            reward = row["total_reward"]
            distance = row["total_path_distance"]
            order = row.get("ordered_target_ids", [])

            print(
                f"  {method:16s} | "
                f"reward={reward:10.4f} | "
                f"path_distance={distance:10.4f} | "
                f"order={order}"
            )

    print("\n========================================================\n")


def main() -> None:
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        无返回值；通过状态变更、文件输出或日志记录体现执行结果。
    """
    args = parse_args()

    project_root = get_project_root()
    config_path = resolve_path(args.config)
    output_path = resolve_path(args.output_path)

    config = load_and_validate_config(config_path)
    apply_command_line_overrides(config, args)

    logger = setup_logger_from_config(config)

    logger.info("=" * 80)
    logger.info("StrikeOrder DQN evaluation started.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Checkpoint type: {args.checkpoint}")
    logger.info(f"Output path: {output_path}")
    logger.info("=" * 80)

    seed = int(get_config_value(config, "experiment.seed", default=42))
    preferred_device = str(
        get_config_value(config, "experiment.device", default="auto")
    )

    set_seed(seed)

    rng = np.random.default_rng(seed)
    device = torch.device(get_device(preferred_device)) if TORCH_AVAILABLE else None

    logger.info(f"Seed: {seed}")
    logger.info(f"Selected device: {device if device is not None else 'no_torch'}")

    data = load_all_data(config)

    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )

    env_config = load_strike_order_env_config(config)
    agent_config = load_strike_order_agent_config(config) if TORCH_AVAILABLE else None
    checkpoint_config = (
        load_strike_order_checkpoint_config(config) if TORCH_AVAILABLE else None
    )

    num_debug_clusters = int(
        get_config_value(
            config,
            "strike_order_dqn.cluster_dataset.num_debug_clusters",
            default=5,
        )
    )
    targets_per_cluster = int(
        get_config_value(
            config,
            "strike_order_dqn.cluster_dataset.targets_per_cluster",
            default=6,
        )
    )

    clusters = build_debug_clusters(
        targets=state.targets,
        num_clusters=num_debug_clusters,
        targets_per_cluster=targets_per_cluster,
    )

    logger.info(
        "Evaluation clusters built: "
        f"num_clusters={len(clusters)}, "
        f"targets_per_cluster={targets_per_cluster}"
    )

    sample_env = StrikeOrderEnv(
        target_cluster=clusters[0],
        start_position=clusters[0].center,
        config=env_config,
    )
    sample_observation = sample_env.reset()

    input_dim = int(sample_observation.vector.shape[0])
    action_dim = sample_env.action_dim

    agent = None
    if TORCH_AVAILABLE:
        network_config = load_strike_order_network_config(
            config=config,
            input_dim=input_dim,
            action_dim=action_dim,
        )

        agent = DQNAgent(
            network_config=network_config,
            agent_config=agent_config,
            device=device,
            seed=seed,
        )

        checkpoint_manager = DQNCheckpointManager(
            config=checkpoint_config,
            project_root=project_root,
        )

        try:
            if args.checkpoint == "best":
                load_result = checkpoint_manager.load_best(
                    agent=agent,
                    load_optimizer=False,
                    map_location=device,
                )
            else:
                load_result = checkpoint_manager.load_latest(
                    agent=agent,
                    load_optimizer=False,
                    map_location=device,
                )
            logger.info(f"Checkpoint loaded: {load_result.path}")
            logger.info(f"Checkpoint metadata: {load_result.metadata}")
        except Exception as exc:
            logger.warning(
                "Could not load DQN checkpoint. Continuing with baselines only. "
                f"Error: {exc}"
            )
            agent = None
    else:
        logger.warning(
            "PyTorch is not installed. Evaluating nearest/random/brute-force "
            "baselines only."
        )

    all_rows: list[dict[str, Any]] = []

    for cluster in clusters:
        rows = evaluate_cluster(
            cluster=cluster,
            env_config=env_config,
            agent=agent,
            rng=rng,
            random_trials=args.random_trials,
        )
        all_rows.extend(rows)

    save_evaluation_rows(
        rows=all_rows,
        output_path=output_path,
    )

    print_evaluation_summary(all_rows)

    logger.info(f"Evaluation CSV saved to: {output_path}")
    logger.info("StrikeOrder DQN evaluation finished successfully.")


if __name__ == "__main__":
    main()
