from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from uav_dynamic_task_allocation.algorithms.rl.dqn.network import (
    DQNNetwork,
    DQNNetworkConfig,
    apply_action_mask,
)
from uav_dynamic_task_allocation.algorithms.rl.dqn.replay_buffer import (
    ExperienceBatch,
)
from uav_dynamic_task_allocation.utils.config import get_config_value


class DQNAgentError(Exception):
    """DQN 智能体动作选择、网络更新和配置解析过程中的自定义错误。"""


@dataclass(frozen=True)
class DQNAgentConfig:
    """
    DQN 智能体配置。

    这个配置控制的是“如何学习”和“如何探索”，不同于 network.py 中的网络结构配置。

    gamma:
        折扣因子，用于计算目标 Q 值：
            target = reward + gamma * max Q(next_state, next_action)

    learning_rate:
        优化器学习率。

    batch_size:
        每次训练从 ReplayBuffer 中采样的样本数量。

    target_update_interval:
        target network 参数同步间隔。DQN 使用 target network 是为了降低训练目标频繁变化带来的不稳定性。

    epsilon_start / epsilon_end / epsilon_decay_steps:
        epsilon-greedy 探索策略参数。训练初期随机探索更多，后期逐渐转向利用 Q 网络。

    gradient_clip_norm:
        梯度裁剪阈值，用于提高训练稳定性。
    """

    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 32
    target_update_interval: int = 100

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000

    gradient_clip_norm: float = 10.0
    optimizer: str = "adam"

    def validate(self) -> None:
        """检查 DQN agent 配置是否合法。"""
        if not 0.0 <= self.gamma <= 1.0:
            raise DQNAgentError("gamma must be in [0, 1].")

        if self.learning_rate <= 0:
            raise DQNAgentError("learning_rate must be positive.")

        if self.batch_size <= 0:
            raise DQNAgentError("batch_size must be positive.")

        if self.target_update_interval <= 0:
            raise DQNAgentError("target_update_interval must be positive.")

        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise DQNAgentError(
                "epsilon values must satisfy "
                "0 <= epsilon_end <= epsilon_start <= 1."
            )

        if self.epsilon_decay_steps <= 0:
            raise DQNAgentError("epsilon_decay_steps must be positive.")

        if self.gradient_clip_norm <= 0:
            raise DQNAgentError("gradient_clip_norm must be positive.")

        if self.optimizer not in {"adam"}:
            raise DQNAgentError(
                f"Unsupported optimizer: {self.optimizer}. "
                "Currently supported optimizer: adam."
            )


@dataclass(frozen=True)
class ActionSelection:
    """
    DQNAgent 的动作选择结果。

    不只返回 action_id，而是同时记录：
    - 当前 epsilon；
    - 是否随机探索；
    - 该动作对应的 Q 值。

    这样后续训练日志中可以分析：
    模型到底是在探索，还是在利用当前 Q 网络。
    """

    action_id: int
    epsilon: float
    is_random: bool
    q_value: float | None = None

    def to_dict(self) -> dict[str, float | int | bool | None]:
        """转换为字典，方便日志输出。"""
        return {
            "action_id": self.action_id,
            "epsilon": self.epsilon,
            "is_random": self.is_random,
            "q_value": self.q_value,
        }


@dataclass(frozen=True)
class DQNUpdateResult:
    """
    一次 DQN 网络更新结果。

    该对象用于记录训练过程中的关键数值：
    - loss；
    - 当前 Q 均值；
    - 目标 Q 均值；
    - 是否同步了 target network。

    后续写 trainer.py 时，这些信息会进入训练日志和 metrics.csv。
    """

    loss: float
    mean_current_q: float
    mean_target_q: float
    target_network_updated: bool

    def to_dict(self) -> dict[str, float | bool]:
        """转换为字典，方便日志输出和保存训练指标。"""
        return {
            "loss": self.loss,
            "mean_current_q": self.mean_current_q,
            "mean_target_q": self.mean_target_q,
            "target_network_updated": self.target_network_updated,
        }


class DQNAgent:
    """
    DQN 智能体。

    该类负责两件核心任务：

    1. 动作选择：
       根据 observation 和 action_mask，使用 epsilon-greedy 策略选择 action_id。

    2. 网络更新：
       从 ReplayBuffer 采样 batch，基于 DQN 目标值更新 policy network。

    它不直接读取 CSV、不直接操作 UAV/Target，也不直接推进环境。
    这些职责分别属于 data、entities、env 模块。
    """

    def __init__(
        self,
        network_config: DQNNetworkConfig,
        agent_config: DQNAgentConfig,
        device: torch.device,
        seed: int | None = None,
    ) -> None:
        agent_config.validate()

        self.network_config = network_config
        self.agent_config = agent_config
        self.device = device

        self.policy_network = DQNNetwork(network_config).to(self.device)
        self.target_network = DQNNetwork(network_config).to(self.device)

        # target network 初始时与 policy network 完全一致。
        self.sync_target_network()

        self.optimizer = self._build_optimizer()
        self.loss_fn = nn.MSELoss()

        self._rng = np.random.default_rng(seed)

        self.total_action_steps = 0
        self.total_update_steps = 0

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        training: bool = True,
    ) -> ActionSelection:
        """
        使用 epsilon-greedy 策略选择动作。

        Args:
            observation:
                当前状态向量，形状为 [input_dim]。
            action_mask:
                当前状态下的动作合法性 mask，形状为 [action_dim]。
                1 表示合法动作，0 表示非法动作。
            training:
                训练时使用 epsilon-greedy；
                评估时不随机探索，直接选择合法动作中 Q 值最大的动作。

        Returns:
            ActionSelection，包含 action_id、epsilon 和是否随机探索。
        """
        self._validate_observation_and_mask(observation, action_mask)

        valid_action_ids = np.where(action_mask > 0)[0]

        if valid_action_ids.size == 0:
            raise DQNAgentError(
                "No valid action available for action selection."
            )

        epsilon = self.get_epsilon() if training else 0.0

        if training and self._rng.random() < epsilon:
            action_id = int(self._rng.choice(valid_action_ids))

            self.total_action_steps += 1

            return ActionSelection(
                action_id=action_id,
                epsilon=epsilon,
                is_random=True,
                q_value=None,
            )

        observation_tensor = torch.from_numpy(observation).float().unsqueeze(0)
        observation_tensor = observation_tensor.to(self.device)

        action_mask_tensor = torch.from_numpy(action_mask).float().unsqueeze(0)
        action_mask_tensor = action_mask_tensor.to(self.device)

        self.policy_network.eval()
        with torch.no_grad():
            q_values = self.policy_network(observation_tensor)
            masked_q_values = apply_action_mask(q_values, action_mask_tensor)
            action_tensor = torch.argmax(masked_q_values, dim=1)

        action_id = int(action_tensor.item())
        q_value = float(q_values[0, action_id].item())

        if training:
            self.total_action_steps += 1

        return ActionSelection(
            action_id=action_id,
            epsilon=epsilon,
            is_random=False,
            q_value=q_value,
        )

    def update(self, batch: ExperienceBatch) -> DQNUpdateResult:
        """
        使用一个 ExperienceBatch 更新 policy network。

        DQN 更新公式：
            current_q = Q(s, a)
            target_q = r + gamma * max_a' Q_target(s', a')   if not done
            target_q = r                                     if done

        其中 next_action_mask 用于保证 max_a' 只在合法动作中计算。
        """
        self.policy_network.train()
        self.target_network.eval()

        states = torch.from_numpy(batch.states).float().to(self.device)
        action_ids = torch.from_numpy(batch.action_ids).long().to(self.device)
        rewards = torch.from_numpy(batch.rewards).float().to(self.device)
        next_states = torch.from_numpy(batch.next_states).float().to(self.device)
        dones = torch.from_numpy(batch.dones).float().to(self.device)

        q_values = self.policy_network(states)

        # gather 根据 action_ids 取出每个样本实际执行动作对应的 Q(s, a)。
        current_q = q_values.gather(
            dim=1,
            index=action_ids.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_max_q = self._masked_max_next_q(
                next_q_values=next_q_values,
                next_action_masks=batch.next_action_masks,
            )

            target_q = rewards + self.agent_config.gamma * (1.0 - dones) * next_max_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=self.agent_config.gradient_clip_norm,
        )

        self.optimizer.step()

        self.total_update_steps += 1

        target_network_updated = False
        if self.total_update_steps % self.agent_config.target_update_interval == 0:
            self.sync_target_network()
            target_network_updated = True

        return DQNUpdateResult(
            loss=float(loss.item()),
            mean_current_q=float(current_q.mean().item()),
            mean_target_q=float(target_q.mean().item()),
            target_network_updated=target_network_updated,
        )

    def get_epsilon(self) -> float:
        """
        根据当前动作步数计算 epsilon。

        使用线性衰减：
            epsilon 从 epsilon_start 逐步下降到 epsilon_end。

        训练初期随机探索更多，后期更多依赖 Q 网络。
        """
        progress = min(
            self.total_action_steps / self.agent_config.epsilon_decay_steps,
            1.0,
        )

        epsilon = (
            self.agent_config.epsilon_start
            + progress
            * (self.agent_config.epsilon_end - self.agent_config.epsilon_start)
        )

        return float(epsilon)

    def sync_target_network(self) -> None:
        """
        将 policy network 参数复制到 target network。

        target network 的作用是稳定目标 Q 值，避免训练目标随着 policy network
        每一步更新而剧烈变化。
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def _masked_max_next_q(
        self,
        next_q_values: torch.Tensor,
        next_action_masks: np.ndarray | None,
    ) -> torch.Tensor:
        """
        计算下一状态的最大 Q 值。

        如果提供 next_action_masks，则只在合法动作中取最大值。
        如果某个样本的下一状态没有合法动作，则该样本的 next_max_q 设为 0，
        这样 target_q 就只等于即时 reward。
        """
        if next_action_masks is None:
            return torch.max(next_q_values, dim=1).values

        masks = torch.from_numpy(next_action_masks).float().to(self.device)

        if masks.shape != next_q_values.shape:
            raise DQNAgentError(
                "next_action_masks shape must match next_q_values shape, "
                f"got masks={tuple(masks.shape)}, "
                f"next_q_values={tuple(next_q_values.shape)}"
            )

        valid_counts = masks.sum(dim=1)
        masked_next_q_values = next_q_values.masked_fill(masks <= 0, -1e9)

        next_max_q = torch.max(masked_next_q_values, dim=1).values

        # 没有合法下一动作的样本，视为终止或不可继续状态，next Q 设为 0。
        next_max_q = torch.where(
            valid_counts > 0,
            next_max_q,
            torch.zeros_like(next_max_q),
        )

        return next_max_q

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """根据配置创建优化器。"""
        if self.agent_config.optimizer == "adam":
            return Adam(
                self.policy_network.parameters(),
                lr=self.agent_config.learning_rate,
            )

        raise DQNAgentError(
            f"Unsupported optimizer: {self.agent_config.optimizer}"
        )

    def _validate_observation_and_mask(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> None:
        """检查 observation 和 action_mask 是否与网络维度一致。"""
        if observation.ndim != 1:
            raise DQNAgentError(
                f"observation must be 1D, got shape={observation.shape}"
            )

        if observation.shape[0] != self.network_config.input_dim:
            raise DQNAgentError(
                "observation dimension mismatch: "
                f"expected={self.network_config.input_dim}, "
                f"got={observation.shape[0]}"
            )

        if action_mask.ndim != 1:
            raise DQNAgentError(
                f"action_mask must be 1D, got shape={action_mask.shape}"
            )

        if action_mask.shape[0] != self.network_config.action_dim:
            raise DQNAgentError(
                "action_mask dimension mismatch: "
                f"expected={self.network_config.action_dim}, "
                f"got={action_mask.shape[0]}"
            )


def load_dqn_agent_config(config: dict[str, Any]) -> DQNAgentConfig:
    """
    从项目总配置中读取 DQNAgent 配置。

    该函数让训练参数从 YAML 进入 Python 对象，避免训练逻辑中出现硬编码。
    """
    agent_config = DQNAgentConfig(
        gamma=float(get_config_value(config, "dqn.agent.gamma", default=0.99)),
        learning_rate=float(
            get_config_value(config, "dqn.agent.learning_rate", default=5e-4)
        ),
        batch_size=int(
            get_config_value(config, "dqn.agent.batch_size", default=32)
        ),
        target_update_interval=int(
            get_config_value(
                config,
                "dqn.agent.target_update_interval",
                default=100,
            )
        ),
        epsilon_start=float(
            get_config_value(config, "dqn.agent.epsilon_start", default=1.0)
        ),
        epsilon_end=float(
            get_config_value(config, "dqn.agent.epsilon_end", default=0.05)
        ),
        epsilon_decay_steps=int(
            get_config_value(
                config,
                "dqn.agent.epsilon_decay_steps",
                default=5000,
            )
        ),
        gradient_clip_norm=float(
            get_config_value(
                config,
                "dqn.agent.gradient_clip_norm",
                default=10.0,
            )
        ),
        optimizer=str(
            get_config_value(config, "dqn.agent.optimizer", default="adam")
        ),
    )

    agent_config.validate()
    return agent_config