from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class ReplayBufferError(Exception):
    """经验回放池相关的自定义错误。"""


@dataclass(frozen=True)
class Transition:
    """
    DQN 的单条状态转移样本。

    论文中的经验回放机制会保存每个时间步的：
        当前状态 s_t
        执行动作 a_t
        即时奖励 r_t
        下一状态 s_{t+1}

    在工程实现中，我们额外保存：
        done：当前 episode 是否结束；
        action_mask：当前状态下哪些动作合法；
        next_action_mask：下一状态下哪些动作合法；
        info：环境返回的调试信息，例如 reward_breakdown。

    保存 action mask 的原因：
    本项目使用固定动作空间。DQN 网络会输出所有动作的 Q 值，
    但不是所有动作在当前状态下都合法。因此训练时需要 next_action_mask
    来计算合法动作中的 max Q(s_{t+1}, a')。
    """

    state: np.ndarray
    action_id: int
    reward: float
    next_state: np.ndarray
    done: bool

    action_mask: np.ndarray | None = None
    next_action_mask: np.ndarray | None = None

    info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        检查 Transition 的基本合法性。

        这里不做过度复杂的业务检查，只确认 DQN 训练必需字段是合理的。
        这样可以尽早发现 observation 维度错误、action_id 类型错误等问题。
        """
        if self.state.ndim != 1:
            raise ReplayBufferError(
                f"state must be a 1D vector, got shape={self.state.shape}"
            )

        if self.next_state.ndim != 1:
            raise ReplayBufferError(
                "next_state must be a 1D vector, "
                f"got shape={self.next_state.shape}"
            )

        if self.state.shape != self.next_state.shape:
            raise ReplayBufferError(
                "state and next_state must have the same shape, "
                f"got state={self.state.shape}, next_state={self.next_state.shape}"
            )

        if self.action_id < 0:
            raise ReplayBufferError(
                f"action_id must be non-negative, got {self.action_id}"
            )

        if self.action_mask is not None and self.action_mask.ndim != 1:
            raise ReplayBufferError(
                "action_mask must be a 1D vector, "
                f"got shape={self.action_mask.shape}"
            )

        if self.next_action_mask is not None and self.next_action_mask.ndim != 1:
            raise ReplayBufferError(
                "next_action_mask must be a 1D vector, "
                f"got shape={self.next_action_mask.shape}"
            )

        if (
            self.action_mask is not None
            and self.next_action_mask is not None
            and self.action_mask.shape != self.next_action_mask.shape
        ):
            raise ReplayBufferError(
                "action_mask and next_action_mask must have the same shape, "
                f"got action_mask={self.action_mask.shape}, "
                f"next_action_mask={self.next_action_mask.shape}"
            )


@dataclass(frozen=True)
class ExperienceBatch:
    """
    从 ReplayBuffer 中采样得到的小批量训练数据。

    这个对象后续会直接喂给 DQN trainer。
    当前仍然使用 NumPy 数组保存，是为了让 ReplayBuffer 不依赖 PyTorch。
    等到正式写 trainer 时，再把这些数组转换成 torch.Tensor。
    """

    states: np.ndarray
    action_ids: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    action_masks: np.ndarray | None = None
    next_action_masks: np.ndarray | None = None

    infos: list[dict[str, Any]] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        """返回 batch 中包含的样本数量。"""
        return int(self.states.shape[0])


class ReplayBuffer:
    """
    DQN 经验回放池。

    该类采用固定容量 FIFO 机制：
    - 当 buffer 未满时，持续追加 transition；
    - 当 buffer 已满时，新样本进入，最旧样本自动移除。

    这样设计对应论文中的经验回放思想：
    存储历史状态转移样本，并在训练时随机采样小批量数据，
    用于降低时间序列样本相关性，提高训练稳定性。
    """

    def __init__(
        self,
        capacity: int,
        seed: int | None = None,
    ) -> None:
        """
        初始化经验回放池。

        Args:
            capacity: 回放池最大容量。
            seed: 随机种子，用于保证采样过程尽可能可复现。
        """
        if capacity <= 0:
            raise ReplayBufferError("Replay buffer capacity must be positive.")

        self.capacity = int(capacity)
        self._buffer: deque[Transition] = deque(maxlen=self.capacity)

        # 使用 NumPy 新版随机数生成器，便于控制采样随机性。
        self._rng = np.random.default_rng(seed)

    def push(self, transition: Transition) -> None:
        """
        向回放池中加入一条 transition。

        push 前会调用 transition.validate()，防止错误样本进入训练数据。
        一旦错误样本进入 ReplayBuffer，后续训练报错会非常难定位，
        所以这里宁愿提前严格检查。
        """
        transition.validate()
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> ExperienceBatch:
        """
        从回放池中随机采样一个 batch。

        DQN 训练时不会直接使用最新的一条 transition，
        而是从历史经验中随机采样多个样本进行更新。
        这正是经验回放提升训练稳定性的核心。
        """
        if batch_size <= 0:
            raise ReplayBufferError("batch_size must be positive.")

        if len(self._buffer) < batch_size:
            raise ReplayBufferError(
                "Not enough samples in replay buffer: "
                f"required={batch_size}, available={len(self._buffer)}"
            )

        indices = self._rng.choice(
            len(self._buffer),
            size=batch_size,
            replace=False,
        )

        transitions = [self._buffer[int(index)] for index in indices]

        return self._collate(transitions)

    def can_sample(self, batch_size: int) -> bool:
        """
        判断当前样本数量是否足够采样一个 batch。

        训练循环中通常会写：
            if replay_buffer.can_sample(batch_size):
                batch = replay_buffer.sample(batch_size)
        """
        return len(self._buffer) >= batch_size

    def clear(self) -> None:
        """清空回放池，通常用于重新开始实验或单元测试。"""
        self._buffer.clear()

    def __len__(self) -> int:
        """返回当前回放池中已有样本数量。"""
        return len(self._buffer)

    def _collate(self, transitions: list[Transition]) -> ExperienceBatch:
        """
        将多条 Transition 合并为一个 ExperienceBatch。

        这个过程类似 PyTorch DataLoader 中的 collate_fn：
        多个单样本字段会被 stack 成 batch 形式。
        """
        states = np.stack(
            [transition.state for transition in transitions],
            axis=0,
        ).astype(np.float32)

        action_ids = np.array(
            [transition.action_id for transition in transitions],
            dtype=np.int64,
        )

        rewards = np.array(
            [transition.reward for transition in transitions],
            dtype=np.float32,
        )

        next_states = np.stack(
            [transition.next_state for transition in transitions],
            axis=0,
        ).astype(np.float32)

        dones = np.array(
            [transition.done for transition in transitions],
            dtype=np.float32,
        )

        action_masks = self._collate_optional_masks(
            [transition.action_mask for transition in transitions]
        )

        next_action_masks = self._collate_optional_masks(
            [transition.next_action_mask for transition in transitions]
        )

        infos = [transition.info for transition in transitions]

        return ExperienceBatch(
            states=states,
            action_ids=action_ids,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            action_masks=action_masks,
            next_action_masks=next_action_masks,
            infos=infos,
        )

    @staticmethod
    def _collate_optional_masks(
        masks: list[np.ndarray | None],
    ) -> np.ndarray | None:
        """
        合并 action mask。

        如果所有 transition 都没有 mask，则返回 None。
        如果部分有 mask、部分没有 mask，这是不允许的，
        因为训练时无法判断哪些样本需要 mask，哪些不需要。
        """
        if all(mask is None for mask in masks):
            return None

        if any(mask is None for mask in masks):
            raise ReplayBufferError(
                "Inconsistent masks: some transitions have masks, "
                "but others do not."
            )

        return np.stack(masks, axis=0).astype(np.float32)