from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from uav_dynamic_task_allocation.algorithms.rl.dqn.agent import (
    DQNAgent,
    DQNUpdateResult,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.checkpoint import (
    DQNCheckpointManager,
    DQNCheckpointMetadata,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.metrics import DQNMetricsWriter
from uav_dynamic_task_allocation.algorithms.rl.dqn.replay_buffer import (
    ReplayBuffer,
    Transition,
)
from uav_dynamic_task_allocation.core.contracts import TargetCluster
from uav_dynamic_task_allocation.core.entities import Position
from uav_dynamic_task_allocation.envs.strike_order_env import (
    StrikeOrderEnv,
    StrikeOrderEnvConfig,
)
from uav_dynamic_task_allocation.utils.config import get_config_value


class StrikeOrderDQNTrainerError(Exception):
    """StrikeOrder DQN 训练流程中的自定义错误。"""


@dataclass(frozen=True)
class StrikeOrderDQNTrainerConfig:
    """
    StrikeOrder DQN 训练配置。

    这个 trainer 专门用于训练目标群内部打击排序 DQN。

    与之前的 DQNTrainer 不同：
    - 之前的 DQNTrainer 面向 DroneBattleEnv；
    - 这里的 StrikeOrderDQNTrainer 面向 StrikeOrderEnv；
    - action_id 表示目标群内部的 target slot；
    - 一个 episode 对应一个目标群内部排序过程。
    """

    num_episodes: int = 100
    max_steps_per_episode: int | str = "auto"

    replay_buffer_capacity: int = 10000
    warmup_steps: int = 100
    train_every_steps: int = 1
    updates_per_train_step: int = 1

    log_interval: int = 1
    save_checkpoint: bool = True

    # random: 每个 episode 随机抽一个目标群训练；
    # round_robin: 按顺序循环目标群。
    cluster_sampling: str = "random"

    def validate(self) -> None:
        """检查训练配置是否合法。"""
        if self.num_episodes <= 0:
            raise StrikeOrderDQNTrainerError("num_episodes must be positive.")

        if isinstance(self.max_steps_per_episode, int):
            if self.max_steps_per_episode <= 0:
                raise StrikeOrderDQNTrainerError(
                    "max_steps_per_episode must be positive."
                )
        elif self.max_steps_per_episode != "auto":
            raise StrikeOrderDQNTrainerError(
                "max_steps_per_episode must be an integer or 'auto'."
            )

        if self.replay_buffer_capacity <= 0:
            raise StrikeOrderDQNTrainerError(
                "replay_buffer_capacity must be positive."
            )

        if self.warmup_steps < 0:
            raise StrikeOrderDQNTrainerError("warmup_steps must be non-negative.")

        if self.train_every_steps <= 0:
            raise StrikeOrderDQNTrainerError("train_every_steps must be positive.")

        if self.updates_per_train_step <= 0:
            raise StrikeOrderDQNTrainerError(
                "updates_per_train_step must be positive."
            )

        if self.log_interval <= 0:
            raise StrikeOrderDQNTrainerError("log_interval must be positive.")

        if self.cluster_sampling not in {"random", "round_robin"}:
            raise StrikeOrderDQNTrainerError(
                "cluster_sampling must be 'random' or 'round_robin'."
            )


@dataclass
class StrikeOrderEpisodeMetrics:
    """
    单个 StrikeOrder episode 的训练指标。

    一个 episode 对应：
        给定一个 TargetCluster，从起点出发，依次选择目标，直到该群目标全部访问完成。
    """

    episode: int
    cluster_id: int

    total_reward: float = 0.0
    num_steps: int = 0
    num_updates: int = 0

    random_action_count: int = 0
    greedy_action_count: int = 0
    invalid_or_no_action_count: int = 0

    losses: list[float] = field(default_factory=list)
    done: bool = False

    ordered_target_ids: list[int] = field(default_factory=list)
    total_path_distance: float = 0.0

    epsilon_after_episode: float | None = None
    best_reward_so_far: float | None = None
    global_env_steps: int = 0
    replay_buffer_size: int = 0

    latest_checkpoint_path: str | None = None
    best_checkpoint_path: str | None = None
    is_best_episode: bool = False

    @property
    def mean_loss(self) -> float | None:
        """返回该 episode 的平均 loss。"""
        if not self.losses:
            return None
        return float(sum(self.losses) / len(self.losses))

    @property
    def random_action_ratio(self) -> float:
        """返回随机动作比例。"""
        total_actions = self.random_action_count + self.greedy_action_count
        if total_actions == 0:
            return 0.0
        return self.random_action_count / total_actions

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，便于写入 metrics CSV / JSONL。"""
        return {
            "episode": self.episode,
            "cluster_id": self.cluster_id,
            "total_reward": self.total_reward,
            "num_steps": self.num_steps,
            "num_updates": self.num_updates,
            "mean_loss": self.mean_loss,
            "random_action_count": self.random_action_count,
            "greedy_action_count": self.greedy_action_count,
            "random_action_ratio": self.random_action_ratio,
            "invalid_or_no_action_count": self.invalid_or_no_action_count,
            "done": self.done,
            "ordered_target_ids": self.ordered_target_ids,
            "total_path_distance": self.total_path_distance,
            "epsilon_after_episode": self.epsilon_after_episode,
            "best_reward_so_far": self.best_reward_so_far,
            "global_env_steps": self.global_env_steps,
            "replay_buffer_size": self.replay_buffer_size,
            "latest_checkpoint_path": self.latest_checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
            "is_best_episode": self.is_best_episode,
        }


@dataclass
class StrikeOrderTrainingResult:
    """StrikeOrder DQN 训练结果。"""

    episode_metrics: list[StrikeOrderEpisodeMetrics]

    @property
    def best_reward(self) -> float | None:
        """返回最高 episode reward。"""
        if not self.episode_metrics:
            return None
        return max(metric.total_reward for metric in self.episode_metrics)

    @property
    def last_reward(self) -> float | None:
        """返回最后一个 episode reward。"""
        if not self.episode_metrics:
            return None
        return self.episode_metrics[-1].total_reward

    @property
    def best_episode(self) -> int | None:
        """返回 best reward 对应的 episode。"""
        if not self.episode_metrics:
            return None

        best_metric = max(
            self.episode_metrics,
            key=lambda metric: metric.total_reward,
        )
        return best_metric.episode


class StrikeOrderDQNTrainer:
    """
    目标群内部打击次序 DQN 训练器。

    它负责训练源代码对齐版 DQN：
        输入一个目标群；
        每一步选择一个未访问目标；
        最终形成该目标群的打击顺序。

    注意：
    这个 trainer 不负责目标筛选、目标分群、资源分配和全流程时间推进。
    这些后面由 target_screening.py、target_clustering.py、
    resource_allocation.py 和 mission_simulator.py 负责。
    """

    def __init__(
        self,
        clusters: list[TargetCluster],
        strike_order_env_config: StrikeOrderEnvConfig,
        agent: DQNAgent,
        replay_buffer: ReplayBuffer,
        trainer_config: StrikeOrderDQNTrainerConfig,
        checkpoint_manager: DQNCheckpointManager | None = None,
        metrics_writer: DQNMetricsWriter | None = None,
        logger: logging.Logger | None = None,
        seed: int | None = None,
    ) -> None:
        trainer_config.validate()

        if not clusters:
            raise StrikeOrderDQNTrainerError("clusters must not be empty.")

        self.clusters = clusters
        self.strike_order_env_config = strike_order_env_config
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.config = trainer_config
        self.checkpoint_manager = checkpoint_manager
        self.metrics_writer = metrics_writer

        self.logger = logger or logging.getLogger(__name__)

        self.global_env_steps = 0
        self.best_reward: float | None = None

        self._rng = np.random.default_rng(seed)

        if self.config.save_checkpoint and self.checkpoint_manager is None:
            raise StrikeOrderDQNTrainerError(
                "save_checkpoint=True, but checkpoint_manager is None."
            )

    def train(self) -> StrikeOrderTrainingResult:
        """执行 StrikeOrder DQN 训练。"""
        metrics_history: list[StrikeOrderEpisodeMetrics] = []

        self.logger.info(
            "StrikeOrder DQN training started: "
            f"num_episodes={self.config.num_episodes}, "
            f"num_clusters={len(self.clusters)}, "
            f"buffer_capacity={self.replay_buffer.capacity}, "
            f"warmup_steps={self.config.warmup_steps}, "
            f"cluster_sampling={self.config.cluster_sampling}"
        )

        for episode in range(1, self.config.num_episodes + 1):
            cluster = self._select_cluster(episode)
            metrics = self._run_episode(
                episode=episode,
                cluster=cluster,
            )

            self._maybe_save_checkpoint(metrics)
            self._maybe_write_metrics(metrics)

            metrics_history.append(metrics)

            if episode % self.config.log_interval == 0:
                self.logger.info(
                    "StrikeOrder episode finished: "
                    f"{metrics.to_dict()}"
                )

        result = StrikeOrderTrainingResult(episode_metrics=metrics_history)

        self.logger.info(
            "StrikeOrder DQN training finished: "
            f"episodes={len(metrics_history)}, "
            f"best_episode={result.best_episode}, "
            f"best_reward={result.best_reward}, "
            f"last_reward={result.last_reward}"
        )

        return result

    def _run_episode(
        self,
        episode: int,
        cluster: TargetCluster,
    ) -> StrikeOrderEpisodeMetrics:
        """
        运行单个目标群排序 episode。
        """
        start_position = self._get_start_position(cluster)

        env = StrikeOrderEnv(
            target_cluster=cluster,
            start_position=start_position,
            config=self.strike_order_env_config,
        )

        observation = env.reset()

        max_steps = self._resolve_max_steps_per_episode(env)

        metrics = StrikeOrderEpisodeMetrics(
            episode=episode,
            cluster_id=cluster.cluster_id,
        )

        for _ in range(max_steps):
            action_mask = observation.action_mask

            if int(action_mask.sum()) == 0:
                metrics.invalid_or_no_action_count += 1
                self.logger.warning(
                    "No valid strike-order action available. "
                    f"episode={episode}, cluster_id={cluster.cluster_id}"
                )
                break

            selection = self.agent.select_action(
                observation=observation.vector,
                action_mask=action_mask,
                training=True,
            )

            if selection.is_random:
                metrics.random_action_count += 1
            else:
                metrics.greedy_action_count += 1

            step_result = env.step(selection.action_id)
            next_observation = step_result.observation

            transition = Transition(
                state=observation.vector,
                action_id=selection.action_id,
                reward=step_result.reward,
                next_state=next_observation.vector,
                done=step_result.done,
                action_mask=action_mask,
                next_action_mask=next_observation.action_mask,
                info={
                    "selection": selection.to_dict(),
                    "env_info": step_result.info,
                    "cluster_id": cluster.cluster_id,
                },
            )

            self.replay_buffer.push(transition)

            metrics.total_reward += step_result.reward
            metrics.num_steps += 1
            self.global_env_steps += 1

            update_results = self._maybe_update_agent()
            for update_result in update_results:
                metrics.losses.append(update_result.loss)
                metrics.num_updates += 1

            observation = next_observation

            if step_result.done:
                metrics.done = True
                break

        plan_dict = env.get_strike_order_plan_dict()
        metrics.ordered_target_ids = plan_dict["ordered_target_ids"]
        metrics.total_path_distance = plan_dict["total_path_distance"]

        return metrics

    def _maybe_update_agent(self) -> list[DQNUpdateResult]:
        """
        根据训练条件决定是否更新 DQNAgent。
        """
        if len(self.replay_buffer) < self.config.warmup_steps:
            return []

        if not self.replay_buffer.can_sample(self.agent.agent_config.batch_size):
            return []

        if self.global_env_steps % self.config.train_every_steps != 0:
            return []

        update_results: list[DQNUpdateResult] = []

        for _ in range(self.config.updates_per_train_step):
            batch = self.replay_buffer.sample(
                self.agent.agent_config.batch_size
            )
            update_result = self.agent.update(batch)
            update_results.append(update_result)

        return update_results

    def _maybe_save_checkpoint(
        self,
        metrics: StrikeOrderEpisodeMetrics,
    ) -> None:
        """
        保存 latest / best checkpoint。
        """
        current_reward = metrics.total_reward

        is_best = (
            self.best_reward is None
            or current_reward > self.best_reward
        )

        if is_best:
            self.best_reward = current_reward
            metrics.is_best_episode = True

        self._finalize_metrics(metrics)

        if not self.config.save_checkpoint:
            return

        if self.checkpoint_manager is None:
            raise StrikeOrderDQNTrainerError(
                "Cannot save checkpoint because checkpoint_manager is None."
            )

        metadata = DQNCheckpointMetadata(
            episode=metrics.episode,
            global_env_steps=self.global_env_steps,
            total_action_steps=self.agent.total_action_steps,
            total_update_steps=self.agent.total_update_steps,
            best_reward=self.best_reward,
            last_reward=current_reward,
            epsilon=self.agent.get_epsilon(),
            extra={
                "task": "strike_order_dqn",
                "episode_metrics": metrics.to_dict(),
                "replay_buffer_size": len(self.replay_buffer),
            },
        )

        latest_path = self.checkpoint_manager.save_latest(
            agent=self.agent,
            metadata=metadata,
        )
        metrics.latest_checkpoint_path = str(latest_path)

        if is_best:
            best_path = self.checkpoint_manager.save_best(
                agent=self.agent,
                metadata=metadata,
            )
            metrics.best_checkpoint_path = str(best_path)

            self.logger.info(
                "StrikeOrder best checkpoint updated: "
                f"episode={metrics.episode}, "
                f"cluster_id={metrics.cluster_id}, "
                f"reward={current_reward:.6f}, "
                f"path={best_path}"
            )

    def _maybe_write_metrics(
        self,
        metrics: StrikeOrderEpisodeMetrics,
    ) -> None:
        """写入 episode 级指标。"""
        if self.metrics_writer is None:
            return

        self.metrics_writer.write_episode(metrics.to_dict())

    def _finalize_metrics(
        self,
        metrics: StrikeOrderEpisodeMetrics,
    ) -> None:
        """补充 episode 结束后的全局信息。"""
        metrics.epsilon_after_episode = self.agent.get_epsilon()
        metrics.best_reward_so_far = self.best_reward
        metrics.global_env_steps = self.global_env_steps
        metrics.replay_buffer_size = len(self.replay_buffer)

    def _select_cluster(self, episode: int) -> TargetCluster:
        """选择当前 episode 使用哪个目标群。"""
        if self.config.cluster_sampling == "round_robin":
            index = (episode - 1) % len(self.clusters)
            return self.clusters[index]

        index = int(self._rng.integers(0, len(self.clusters)))
        return self.clusters[index]

    def _get_start_position(self, cluster: TargetCluster) -> Position:
        """
        获取目标群排序起点。

        当前先使用 cluster.center。
        这代表 DQN 训练阶段默认 UAV 小组已经进入目标群任务区。
        后面 MissionSimulator 中可以传入真实 UAV 小组当前位置。
        """
        return cluster.center

    def _resolve_max_steps_per_episode(self, env: StrikeOrderEnv) -> int:
        """
        解析单 episode 最大步数。

        auto 模式下，最多走真实目标数量步。
        因为每一步选择一个未访问目标，全部访问后 done=True。
        """
        if self.config.max_steps_per_episode == "auto":
            return env.num_real_targets

        return int(self.config.max_steps_per_episode)


def load_strike_order_dqn_trainer_config(
    config: dict[str, Any],
) -> StrikeOrderDQNTrainerConfig:
    """
    从项目总配置中读取 StrikeOrder DQN trainer 配置。
    """
    prefix = "strike_order_dqn.trainer"

    raw_max_steps = get_config_value(
        config,
        f"{prefix}.max_steps_per_episode",
        default="auto",
    )

    if isinstance(raw_max_steps, str) and raw_max_steps != "auto":
        try:
            max_steps_per_episode: int | str = int(raw_max_steps)
        except ValueError as exc:
            raise StrikeOrderDQNTrainerError(
                "max_steps_per_episode must be 'auto' or an integer."
            ) from exc
    else:
        max_steps_per_episode = raw_max_steps

    trainer_config = StrikeOrderDQNTrainerConfig(
        num_episodes=int(
            get_config_value(config, f"{prefix}.num_episodes", default=100)
        ),
        max_steps_per_episode=max_steps_per_episode,
        replay_buffer_capacity=int(
            get_config_value(
                config,
                f"{prefix}.replay_buffer_capacity",
                default=10000,
            )
        ),
        warmup_steps=int(
            get_config_value(config, f"{prefix}.warmup_steps", default=100)
        ),
        train_every_steps=int(
            get_config_value(config, f"{prefix}.train_every_steps", default=1)
        ),
        updates_per_train_step=int(
            get_config_value(
                config,
                f"{prefix}.updates_per_train_step",
                default=1,
            )
        ),
        log_interval=int(
            get_config_value(config, f"{prefix}.log_interval", default=1)
        ),
        save_checkpoint=bool(
            get_config_value(
                config,
                f"{prefix}.save_checkpoint",
                default=True,
            )
        ),
        cluster_sampling=str(
            get_config_value(
                config,
                f"{prefix}.cluster_sampling",
                default="random",
            )
        ),
    )

    trainer_config.validate()
    return trainer_config