from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from uav_dynamic_task_allocation.algorithms.rl.dqn.agent import (
    DQNAgent,
    DQNUpdateResult,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.checkpoint import (
    DQNCheckpointManager,
    DQNCheckpointMetadata,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.metrics import (
    DQNMetricsWriter,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.replay_buffer import (
    ReplayBuffer,
    Transition,
)
from uav_dynamic_task_allocation.envs.action_space import (
    FixedUAVTargetActionSpace,
)
from uav_dynamic_task_allocation.envs.drone_battle_env import DroneBattleEnv
from uav_dynamic_task_allocation.envs.observation import ObservationBuilder
from uav_dynamic_task_allocation.utils.config import get_config_value


class DQNTrainerError(Exception):
    """DQN 训练流程中的自定义错误。"""


@dataclass(frozen=True)
class DQNTrainerConfig:
    """
    DQNTrainer 训练配置。

    这个配置控制完整训练流程，而不是单次网络更新。

    num_episodes:
        总训练 episode 数。

    max_steps_per_episode:
        每个 episode 最多执行多少步。可以是整数，也可以使用 "auto"，
        让 trainer 自动使用 env_config.max_steps。

    replay_buffer_capacity:
        经验回放池容量。

    warmup_steps:
        replay buffer 中至少积累多少条经验后，才开始训练 Q 网络。
        这样可以避免训练初期样本太少导致网络更新非常不稳定。

    train_every_steps:
        每隔多少个环境 step 执行一次训练。

    updates_per_train_step:
        每次触发训练时，执行几次 agent.update()。

    log_interval:
        每隔多少个 episode 输出一次训练统计。

    save_checkpoint:
        是否在训练过程中自动保存 latest.pt 和 best.pt。
    """

    num_episodes: int = 5
    max_steps_per_episode: int | str = "auto"

    replay_buffer_capacity: int = 10000
    warmup_steps: int = 32
    train_every_steps: int = 1
    updates_per_train_step: int = 1

    log_interval: int = 1
    save_checkpoint: bool = False

    def validate(self) -> None:
        """检查训练配置是否合法。"""
        if self.num_episodes <= 0:
            raise DQNTrainerError("num_episodes must be positive.")

        if isinstance(self.max_steps_per_episode, int):
            if self.max_steps_per_episode <= 0:
                raise DQNTrainerError(
                    "max_steps_per_episode must be positive."
                )
        elif self.max_steps_per_episode != "auto":
            raise DQNTrainerError(
                "max_steps_per_episode must be an integer or 'auto'."
            )

        if self.replay_buffer_capacity <= 0:
            raise DQNTrainerError("replay_buffer_capacity must be positive.")

        if self.warmup_steps < 0:
            raise DQNTrainerError("warmup_steps must be non-negative.")

        if self.train_every_steps <= 0:
            raise DQNTrainerError("train_every_steps must be positive.")

        if self.updates_per_train_step <= 0:
            raise DQNTrainerError("updates_per_train_step must be positive.")

        if self.log_interval <= 0:
            raise DQNTrainerError("log_interval must be positive.")


@dataclass
class EpisodeMetrics:
    """
    单个 episode 的训练统计信息。

    这些指标用于判断训练是否正常：
    - total_reward 是否逐渐上升；
    - mean_loss 是否异常；
    - random_action_ratio 是否随 epsilon 衰减下降；
    - invalid_or_no_action_count 是否频繁出现；
    - latest_checkpoint_path / best_checkpoint_path 是否正确生成。
    """

    episode: int
    total_reward: float = 0.0
    num_steps: int = 0
    num_updates: int = 0

    random_action_count: int = 0
    greedy_action_count: int = 0
    invalid_or_no_action_count: int = 0

    losses: list[float] = field(default_factory=list)
    done: bool = False

    latest_checkpoint_path: str | None = None
    best_checkpoint_path: str | None = None
    is_best_episode: bool = False

    epsilon_after_episode: float | None = None
    best_reward_so_far: float | None = None
    global_env_steps: int = 0
    replay_buffer_size: int = 0

    @property
    def mean_loss(self) -> float | None:
        """返回该 episode 的平均 loss。如果没有更新，则返回 None。"""
        if not self.losses:
            return None
        return float(sum(self.losses) / len(self.losses))

    @property
    def random_action_ratio(self) -> float:
        """返回随机探索动作比例。"""
        total_actions = self.random_action_count + self.greedy_action_count
        if total_actions == 0:
            return 0.0
        return self.random_action_count / total_actions

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，方便日志输出和后续保存为 CSV。"""
        return {
            "episode": self.episode,
            "total_reward": self.total_reward,
            "num_steps": self.num_steps,
            "num_updates": self.num_updates,
            "mean_loss": self.mean_loss,
            "random_action_count": self.random_action_count,
            "greedy_action_count": self.greedy_action_count,
            "random_action_ratio": self.random_action_ratio,
            "invalid_or_no_action_count": self.invalid_or_no_action_count,
            "done": self.done,
            "is_best_episode": self.is_best_episode,
            "epsilon_after_episode": self.epsilon_after_episode,
            "best_reward_so_far": self.best_reward_so_far,
            "global_env_steps": self.global_env_steps,
            "replay_buffer_size": self.replay_buffer_size,
            "latest_checkpoint_path": self.latest_checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
        }


@dataclass
class TrainingResult:
    """
    DQN 训练结果。

    当前保存 episode 级别指标。后续还会继续扩展：
    - metrics.csv 路径；
    - TensorBoard 路径；
    - checkpoint 路径；
    - 训练耗时；
    - evaluation 指标。
    """

    episode_metrics: list[EpisodeMetrics]

    @property
    def best_reward(self) -> float | None:
        """返回训练过程中最高 episode reward。"""
        if not self.episode_metrics:
            return None
        return max(metric.total_reward for metric in self.episode_metrics)

    @property
    def last_reward(self) -> float | None:
        """返回最后一个 episode 的 reward。"""
        if not self.episode_metrics:
            return None
        return self.episode_metrics[-1].total_reward

    @property
    def best_episode(self) -> int | None:
        """返回 best reward 对应的 episode 编号。"""
        if not self.episode_metrics:
            return None

        best_metric = max(
            self.episode_metrics,
            key=lambda metric: metric.total_reward,
        )
        return best_metric.episode


class DQNTrainer:
    """
    DQN 训练器。

    该类负责完整训练流程：
    1. reset 环境；
    2. 构造 observation 和 action mask；
    3. 调用 agent 选择动作；
    4. 解码 action_id 并交给环境执行；
    5. 构造 transition 并写入 replay buffer；
    6. 在条件满足时更新 DQN 网络；
    7. 记录 episode 训练指标；
    8. 根据配置保存 latest / best checkpoint。

    Trainer 不负责 CSV 读取、reward 内部公式、网络结构定义。
    它只负责协调已经构建好的模块。
    """

    def __init__(
        self,
        env: DroneBattleEnv,
        observation_builder: ObservationBuilder,
        action_space: FixedUAVTargetActionSpace,
        agent: DQNAgent,
        replay_buffer: ReplayBuffer,
        trainer_config: DQNTrainerConfig,
        checkpoint_manager: DQNCheckpointManager | None = None,
        metrics_writer: DQNMetricsWriter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        trainer_config.validate()

        self.env = env
        self.observation_builder = observation_builder
        self.action_space = action_space
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.config = trainer_config
        self.checkpoint_manager = checkpoint_manager
        self.metrics_writer = metrics_writer

        self.logger = logger or logging.getLogger(__name__)

        self.global_env_steps = 0
        self.best_reward: float | None = None

        if self.config.save_checkpoint and self.checkpoint_manager is None:
            raise DQNTrainerError(
                "trainer_config.save_checkpoint=True, but checkpoint_manager is None."
            )

    def train(self) -> TrainingResult:
        """
        执行完整 DQN 训练流程。

        当前版本已经具备完整训练骨架和 checkpoint 保存能力。
        后续会继续增加：
        - metrics.csv；
        - TensorBoard；
        - evaluation；
        - resume training；
        - early stopping。
        """
        metrics_history: list[EpisodeMetrics] = []

        max_steps = self._resolve_max_steps_per_episode()

        self.logger.info(
            "DQN training started: "
            f"num_episodes={self.config.num_episodes}, "
            f"max_steps_per_episode={max_steps}, "
            f"buffer_capacity={self.replay_buffer.capacity}, "
            f"warmup_steps={self.config.warmup_steps}, "
            f"save_checkpoint={self.config.save_checkpoint}"
        )

        for episode in range(1, self.config.num_episodes + 1):
            episode_metrics = self._run_episode(
                episode=episode,
                max_steps=max_steps,
            )

            self._maybe_save_checkpoint(episode_metrics)
            self._maybe_write_metrics(episode_metrics)
            metrics_history.append(episode_metrics)

            if episode % self.config.log_interval == 0:
                self.logger.info(
                    "Episode finished: "
                    f"{episode_metrics.to_dict()}"
                )

        result = TrainingResult(episode_metrics=metrics_history)

        self.logger.info(
            "DQN training finished: "
            f"episodes={len(metrics_history)}, "
            f"best_episode={result.best_episode}, "
            f"best_reward={result.best_reward}, "
            f"last_reward={result.last_reward}"
        )

        return result

    def _run_episode(
        self,
        episode: int,
        max_steps: int,
    ) -> EpisodeMetrics:
        """
        运行单个 episode。

        一个 episode 内不断执行：
            observation -> select_action -> env.step -> replay_buffer.push -> update
        直到 done=True 或达到 max_steps。
        """
        self.env.reset()

        metrics = EpisodeMetrics(episode=episode)

        for _ in range(max_steps):
            state = self.env.get_state_copy()

            observation = self.observation_builder.build(state)
            action_mask = self.action_space.build_action_mask(state)

            if int(action_mask.sum()) == 0:
                metrics.invalid_or_no_action_count += 1
                self.logger.warning(
                    "No valid action available. "
                    "Ending current episode early. "
                    "Consider setting action_space.require_reachable=false "
                    "during early pipeline testing."
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

            decoded_action = self.action_space.decode_action_id(
                state=state,
                action_id=selection.action_id,
            )
            env_action = decoded_action.to_env_action()

            step_result = self.env.step(env_action)

            next_state = self.env.get_state_copy()
            next_observation = self.observation_builder.build(next_state)
            next_action_mask = self.action_space.build_action_mask(next_state)

            transition = Transition(
                state=observation.vector,
                action_id=selection.action_id,
                reward=step_result.reward,
                next_state=next_observation.vector,
                done=step_result.done,
                action_mask=action_mask,
                next_action_mask=next_action_mask,
                info={
                    "selection": selection.to_dict(),
                    "decoded_action": decoded_action.to_dict(),
                    "env_info": step_result.info,
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

            if step_result.done:
                metrics.done = True
                break

        return metrics

    def _maybe_update_agent(self) -> list[DQNUpdateResult]:
        """
        根据训练条件决定是否更新 DQNAgent。

        更新条件：
        1. replay buffer 中样本数达到 warmup_steps；
        2. replay buffer 中样本数达到 batch_size；
        3. 当前 global_env_steps 满足 train_every_steps。
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

    def _maybe_save_checkpoint(self, episode_metrics: EpisodeMetrics) -> None:
        """
        根据配置保存 latest / best checkpoint。

        保存策略：
        1. 每个 episode 结束后更新 best reward 状态；
        2. 如果开启 checkpoint，每个 episode 保存 latest.pt；
        3. 如果当前 total_reward 超过历史 best_reward，保存 best.pt。

        注意：
        即使 save_checkpoint=False，也仍然会更新 best_reward_so_far，
        因为训练指标保存也需要这个信息。
        """
        current_reward = episode_metrics.total_reward

        is_best = (
                self.best_reward is None
                or current_reward > self.best_reward
        )

        if is_best:
            self.best_reward = current_reward
            episode_metrics.is_best_episode = True

        self._finalize_episode_metrics(episode_metrics)

        if not self.config.save_checkpoint:
            return

        if self.checkpoint_manager is None:
            raise DQNTrainerError(
                "Cannot save checkpoint because checkpoint_manager is None."
            )

        metadata = DQNCheckpointMetadata(
            episode=episode_metrics.episode,
            global_env_steps=self.global_env_steps,
            total_action_steps=self.agent.total_action_steps,
            total_update_steps=self.agent.total_update_steps,
            best_reward=self.best_reward,
            last_reward=current_reward,
            epsilon=self.agent.get_epsilon(),
            extra={
                "episode_metrics": episode_metrics.to_dict(),
                "replay_buffer_size": len(self.replay_buffer),
            },
        )

        latest_path = self.checkpoint_manager.save_latest(
            agent=self.agent,
            metadata=metadata,
        )
        episode_metrics.latest_checkpoint_path = str(latest_path)

        if is_best:
            best_path = self.checkpoint_manager.save_best(
                agent=self.agent,
                metadata=metadata,
            )
            episode_metrics.best_checkpoint_path = str(best_path)

            self.logger.info(
                "Best checkpoint updated: "
                f"episode={episode_metrics.episode}, "
                f"reward={current_reward:.6f}, "
                f"path={best_path}"
            )

    def _finalize_episode_metrics(self, episode_metrics: EpisodeMetrics) -> None:
        """
        补充 episode 结束后的全局训练信息。

        这些信息不是单步环境交互直接产生的，
        而是在 episode 结束时根据 agent、trainer、replay buffer 的状态整理出来。
        """
        episode_metrics.epsilon_after_episode = self.agent.get_epsilon()
        episode_metrics.best_reward_so_far = self.best_reward
        episode_metrics.global_env_steps = self.global_env_steps
        episode_metrics.replay_buffer_size = len(self.replay_buffer)

    def _maybe_write_metrics(self, episode_metrics: EpisodeMetrics) -> None:
        """
        如果配置了 metrics_writer，则把当前 episode 指标写入 CSV / JSONL。
        """
        if self.metrics_writer is None:
            return

        self.metrics_writer.write_episode(episode_metrics.to_dict())

    def _resolve_max_steps_per_episode(self) -> int:
        """
        解析每个 episode 的最大步数。

        如果配置为 auto，则使用 EnvConfig.max_steps；
        否则使用配置文件中的整数值。
        """
        if self.config.max_steps_per_episode == "auto":
            return int(self.env.env_config.max_steps)

        return int(self.config.max_steps_per_episode)


def load_dqn_trainer_config(config: dict[str, Any]) -> DQNTrainerConfig:
    """
    从项目总配置中读取 DQNTrainer 配置。

    该函数让训练循环参数从 YAML 进入 Python，
    避免在 trainer.py 中硬编码 episode 数、buffer 容量和 warmup 步数。
    """
    raw_max_steps = get_config_value(
        config,
        "dqn.trainer.max_steps_per_episode",
        default="auto",
    )

    if isinstance(raw_max_steps, str) and raw_max_steps != "auto":
        try:
            max_steps_per_episode: int | str = int(raw_max_steps)
        except ValueError as exc:
            raise DQNTrainerError(
                "dqn.trainer.max_steps_per_episode must be 'auto' or an integer."
            ) from exc
    else:
        max_steps_per_episode = raw_max_steps

    trainer_config = DQNTrainerConfig(
        num_episodes=int(
            get_config_value(
                config,
                "dqn.trainer.num_episodes",
                default=5,
            )
        ),
        max_steps_per_episode=max_steps_per_episode,
        replay_buffer_capacity=int(
            get_config_value(
                config,
                "dqn.trainer.replay_buffer_capacity",
                default=10000,
            )
        ),
        warmup_steps=int(
            get_config_value(
                config,
                "dqn.trainer.warmup_steps",
                default=32,
            )
        ),
        train_every_steps=int(
            get_config_value(
                config,
                "dqn.trainer.train_every_steps",
                default=1,
            )
        ),
        updates_per_train_step=int(
            get_config_value(
                config,
                "dqn.trainer.updates_per_train_step",
                default=1,
            )
        ),
        log_interval=int(
            get_config_value(
                config,
                "dqn.trainer.log_interval",
                default=1,
            )
        ),
        save_checkpoint=bool(
            get_config_value(
                config,
                "dqn.trainer.save_checkpoint",
                default=False,
            )
        ),
    )

    trainer_config.validate()
    return trainer_config