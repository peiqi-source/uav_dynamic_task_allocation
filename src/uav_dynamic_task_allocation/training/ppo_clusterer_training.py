"""Training pipeline for real-data PPO dynamic target regrouping."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from uav_dynamic_task_allocation.algorithms.rl.ppo.agent import (
    PPOAgent,
    PPOAgentConfig,
)
from uav_dynamic_task_allocation.algorithms.rl.ppo.buffer import PPORolloutBuffer
from uav_dynamic_task_allocation.algorithms.rl.ppo.checkpoint import save_ppo_checkpoint
from uav_dynamic_task_allocation.algorithms.rl.ppo.metrics import write_ppo_metrics
from uav_dynamic_task_allocation.algorithms.rl.ppo.network import PPONetworkConfig, torch
from uav_dynamic_task_allocation.algorithms.rl.ppo.visualization import (
    plot_training_losses,
    plot_training_reward,
    plot_training_score,
)
from uav_dynamic_task_allocation.allocation.pso_clusterer import PSOClustererConfig
from uav_dynamic_task_allocation.core.entities import build_battlefield_state
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.envs.target_grouping_env import (
    TargetGroupingEnv,
    TargetGroupingEnvConfig,
)
from uav_dynamic_task_allocation.preprocessing.destroy_target_selection import (
    DestroyTargetSelector,
    load_destroy_target_selection_config,
)
from uav_dynamic_task_allocation.preprocessing.target_screening import (
    TargetScreener,
    load_target_screening_config,
)
from uav_dynamic_task_allocation.training.ppo_scenario_generator import (
    PPOGroupingScenarioGenerator,
    PPOGroupingScenarioGeneratorConfig,
)
from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    get_project_root,
    load_and_validate_config,
    resolve_path,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config
from uav_dynamic_task_allocation.utils.seed import set_seed


def run_ppo_clusterer_training(config_path: str | Path) -> None:
    """Train PPO target regrouping from real CSV-derived dynamic scenarios."""
    project_root = get_project_root()
    config_path = _resolve_config_path(config_path, project_root)
    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    seed = int(get_config_value(config, "ppo_clusterer_training.random_seed", 42))
    set_seed(seed)

    training_config = _training_section(config)
    checkpoint_dir = resolve_path(
        training_config.get("checkpoint_dir", "checkpoints/ppo_clusterer"),
        project_root=project_root,
    )
    output_paths = _output_paths(config, project_root)

    logger.info("=" * 80)
    logger.info("PPO clusterer training started.")
    logger.info("Config path: %s", config_path)
    logger.info("Checkpoint dir: %s", checkpoint_dir)
    logger.info("=" * 80)

    data = load_all_data(config)
    state = build_battlefield_state(
        uav_df=data["uav"],
        target_df=data["target"],
    )
    screened_set = TargetScreener(load_target_screening_config(config)).screen(
        state.targets
    )
    selection_result = DestroyTargetSelector(
        load_destroy_target_selection_config(config)
    ).select(screened_set)
    selected_set = selection_result.to_screened_target_set()

    num_clusters = int(training_config.get("num_clusters", 3))
    feature_names = tuple(training_config.get("feature_names", ["x", "y"]))
    env_config = _env_config(config, num_clusters=num_clusters)
    pso_config = _pso_config(config, seed)

    scenario_generator = PPOGroupingScenarioGenerator(
        destroy_targets=selection_result.destroy_targets,
        non_destroy_targets=selection_result.non_destroy_targets,
        target_scores=selected_set.target_scores,
        config=PPOGroupingScenarioGeneratorConfig(
            feature_names=feature_names,
            num_clusters=num_clusters,
            random_seed=seed,
            probabilities=get_config_value(
                config,
                "ppo_clusterer_training.scenario_generator.probabilities",
                None,
            ),
            target_removed=get_config_value(
                config,
                "ppo_clusterer_training.scenario_generator.target_removed",
                {},
            ),
            target_added=get_config_value(
                config,
                "ppo_clusterer_training.scenario_generator.target_added",
                {},
            ),
            target_value_changed=get_config_value(
                config,
                "ppo_clusterer_training.scenario_generator.target_value_changed",
                {},
            ),
            mixed=get_config_value(
                config,
                "ppo_clusterer_training.scenario_generator.mixed",
                {},
            ),
            pso_config=pso_config,
        ),
    )

    sample_scenario = scenario_generator.generate_episode_scenario(0)
    sample_env = _build_env(sample_scenario, env_config)
    observation_dim = sample_env.observation_dim
    action_dim = sample_env.action_dim
    hidden_dims = tuple(
        int(value)
        for value in get_config_value(
            config,
            "ppo_clusterer_training.network.hidden_dims",
            [128, 128],
        )
    )

    if torch is None:
        rows = [_no_torch_row(sample_scenario, sample_env)]
        write_ppo_metrics(rows, output_paths["training_metrics_csv"])
        _save_training_figures(output_paths)
        save_ppo_checkpoint(
            checkpoint_dir,
            metadata={
                "algorithm": "ppo_clusterer",
                "status": "no_torch_fallback",
                "reason": "PyTorch is not available.",
                "num_clusters": num_clusters,
                "feature_names": list(feature_names),
                "training_metrics_csv": str(output_paths["training_metrics_csv"]),
            },
        )
        logger.warning("PyTorch is not available; saved no_torch_fallback metadata.")
        return

    agent = PPOAgent(
        network_config=PPONetworkConfig(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ),
        agent_config=_agent_config(config),
        seed=seed,
        device=str(training_config.get("device", "cpu")),
    )

    max_episodes = int(training_config.get("max_episodes", 500))
    rows: list[dict[str, Any]] = []

    for episode in range(1, max_episodes + 1):
        scenario = scenario_generator.generate_episode_scenario(episode)
        env = _build_env(scenario, env_config)
        observation = env.reset()
        buffer = PPORolloutBuffer()
        total_reward = 0.0
        done = False

        while not done:
            action_mask = env.action_mask()
            if not action_mask.any():
                break
            action, log_prob, value = agent.select_action(observation, action_mask)
            next_observation, reward, done, _ = env.step_index(action)
            buffer.add(
                observation=observation,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
                action_mask=action_mask,
            )
            total_reward += reward
            observation = next_observation

        update_metrics = agent.update(buffer)
        env_metrics = env.get_metrics()
        row = {
            "episode": episode,
            "scenario_id": scenario.scenario_id,
            "event_type": scenario.event_type,
            "num_targets": len(scenario.targets),
            "total_reward": total_reward,
            "final_score": env_metrics["final_score"],
            "count_balance": env_metrics["count_balance"],
            "compactness": env_metrics["compactness"],
            "workload_balance": env_metrics["workload_balance"],
            "count_std": env_metrics["count_std"],
            "workload_std": env_metrics["workload_std"],
            "empty_cluster_count": env_metrics["empty_cluster_count"],
            "policy_loss": update_metrics["policy_loss"],
            "value_loss": update_metrics["value_loss"],
            "entropy": update_metrics["entropy"],
        }
        rows.append(row)

        if episode == 1 or episode % 25 == 0 or episode == max_episodes:
            logger.info(
                "PPO episode=%s event=%s targets=%s reward=%.4f score=%.4f",
                episode,
                scenario.event_type,
                len(scenario.targets),
                total_reward,
                env_metrics["final_score"],
            )

    write_ppo_metrics(rows, output_paths["training_metrics_csv"])
    _save_training_figures(output_paths)

    save_ppo_checkpoint(
        checkpoint_dir,
        model=agent.network,
        metadata={
            "algorithm": "ppo_clusterer",
            "status": "trained",
            "random_seed": seed,
            "num_clusters": num_clusters,
            "feature_names": list(feature_names),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dims": list(hidden_dims),
            "max_targets_per_cluster": env_config.max_targets_per_cluster,
            "env_config": env_config.__dict__,
            "episodes": len(rows),
            "training_metrics_csv": str(output_paths["training_metrics_csv"]),
            "last_metrics": rows[-1] if rows else {},
        },
    )
    logger.info("PPO checkpoint saved to: %s", checkpoint_dir)
    logger.info("PPO training metrics saved to: %s", output_paths["training_metrics_csv"])


def _resolve_config_path(config_path: str | Path, project_root: Path) -> Path:
    path = Path(config_path)
    return path if path.is_absolute() else project_root / path


def _training_section(config: dict[str, Any]) -> dict[str, Any]:
    return dict(get_config_value(config, "ppo_clusterer_training", {}))


def _agent_config(config: dict[str, Any]) -> PPOAgentConfig:
    prefix = "ppo_clusterer_training.agent"
    return PPOAgentConfig(
        learning_rate=float(get_config_value(config, f"{prefix}.learning_rate", 3e-4)),
        gamma=float(get_config_value(config, f"{prefix}.gamma", 0.99)),
        gae_lambda=float(get_config_value(config, f"{prefix}.gae_lambda", 0.95)),
        clip_epsilon=float(get_config_value(config, f"{prefix}.clip_epsilon", 0.2)),
        entropy_coef=float(get_config_value(config, f"{prefix}.entropy_coef", 0.01)),
        value_loss_coef=float(
            get_config_value(config, f"{prefix}.value_loss_coef", 0.5)
        ),
        batch_size=int(get_config_value(config, f"{prefix}.batch_size", 64)),
        update_epochs=int(get_config_value(config, f"{prefix}.update_epochs", 4)),
    )


def _env_config(config: dict[str, Any], num_clusters: int) -> TargetGroupingEnvConfig:
    prefix = "ppo_clusterer_training.env"
    training = _training_section(config)
    return TargetGroupingEnvConfig(
        num_clusters=num_clusters,
        max_steps=int(training.get("max_steps", 30)),
        max_targets_per_cluster=int(
            get_config_value(config, f"{prefix}.max_targets_per_cluster", 64)
        ),
        illegal_action_penalty=float(
            get_config_value(config, f"{prefix}.illegal_action_penalty", -1.0)
        ),
        count_balance_weight=float(
            get_config_value(config, f"{prefix}.count_balance_weight", 1.0)
        ),
        compactness_weight=float(
            get_config_value(config, f"{prefix}.compactness_weight", 1.0)
        ),
        workload_balance_weight=float(
            get_config_value(config, f"{prefix}.workload_balance_weight", 1.0)
        ),
        empty_cluster_penalty=float(
            get_config_value(config, f"{prefix}.empty_cluster_penalty", -1.0)
        ),
        move_penalty=float(get_config_value(config, f"{prefix}.move_penalty", -0.01)),
        improvement_bonus=float(
            get_config_value(config, f"{prefix}.improvement_bonus", 0.05)
        ),
        patience=int(get_config_value(config, f"{prefix}.patience", 8)),
    )


def _pso_config(config: dict[str, Any], seed: int) -> PSOClustererConfig:
    prefix = "target_clustering.pso"
    return PSOClustererConfig(
        num_particles=int(get_config_value(config, f"{prefix}.num_particles", 30)),
        max_iter=int(get_config_value(config, f"{prefix}.max_iter", 100)),
        inertia_weight=float(get_config_value(config, f"{prefix}.inertia_weight", 0.7)),
        cognitive_weight=float(get_config_value(config, f"{prefix}.cognitive_weight", 1.5)),
        social_weight=float(get_config_value(config, f"{prefix}.social_weight", 1.5)),
        compactness_weight=float(get_config_value(config, f"{prefix}.compactness_weight", 1.0)),
        balance_weight=float(get_config_value(config, f"{prefix}.balance_weight", 0.1)),
        value_weight=float(get_config_value(config, f"{prefix}.value_weight", 0.2)),
        defense_weight=float(get_config_value(config, f"{prefix}.defense_weight", 0.2)),
        convergence_threshold=float(
            get_config_value(config, f"{prefix}.convergence_threshold", 1e-6)
        ),
        random_seed=int(get_config_value(config, f"{prefix}.random_seed", seed)),
        initialize_with_high_value_targets=bool(
            get_config_value(
                config,
                f"{prefix}.initialize_with_high_value_targets",
                True,
            )
        ),
    )


def _build_env(scenario, env_config: TargetGroupingEnvConfig) -> TargetGroupingEnv:
    return TargetGroupingEnv(
        features=scenario.features,
        labels=scenario.initial_labels,
        config=env_config,
        target_values=scenario.target_values,
        target_defenses=scenario.target_defenses,
        metadata=scenario.metadata,
    )


def _no_torch_row(scenario, env: TargetGroupingEnv) -> dict[str, Any]:
    metrics = env.get_metrics()
    return {
        "episode": 0,
        "scenario_id": scenario.scenario_id,
        "event_type": scenario.event_type,
        "num_targets": len(scenario.targets),
        "total_reward": 0.0,
        "final_score": metrics["final_score"],
        "count_balance": metrics["count_balance"],
        "compactness": metrics["compactness"],
        "workload_balance": metrics["workload_balance"],
        "count_std": metrics["count_std"],
        "workload_std": metrics["workload_std"],
        "empty_cluster_count": metrics["empty_cluster_count"],
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
    }


def _output_paths(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    prefix = "ppo_clusterer_training.output"
    return {
        "training_metrics_csv": resolve_path(
            get_config_value(
                config,
                f"{prefix}.training_metrics_csv",
                "outputs/metrics/ppo_clusterer_training_metrics.csv",
            ),
            project_root=project_root,
        ),
        "reward_figure": resolve_path(
            get_config_value(
                config,
                f"{prefix}.reward_figure",
                "outputs/evaluation/figures/ppo_training_reward.png",
            ),
            project_root=project_root,
        ),
        "score_figure": resolve_path(
            get_config_value(
                config,
                f"{prefix}.score_figure",
                "outputs/evaluation/figures/ppo_training_score.png",
            ),
            project_root=project_root,
        ),
        "losses_figure": resolve_path(
            get_config_value(
                config,
                f"{prefix}.losses_figure",
                "outputs/evaluation/figures/ppo_training_losses.png",
            ),
            project_root=project_root,
        ),
    }


def _save_training_figures(output_paths: dict[str, Path]) -> None:
    metrics_csv = output_paths["training_metrics_csv"]
    plot_training_reward(metrics_csv, output_paths["reward_figure"])
    plot_training_score(metrics_csv, output_paths["score_figure"])
    plot_training_losses(metrics_csv, output_paths["losses_figure"])
