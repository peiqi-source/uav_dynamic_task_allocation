"""PPO 算法模块中的经验缓冲区实现。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PPORolloutBuffer:
    """Rollout buffer for clipped-objective PPO updates."""

    # observations: observations 数据。
    observations: list[np.ndarray] = field(default_factory=list)
    # actions: 动作集合。
    actions: list[int] = field(default_factory=list)
    # log_probs: 日志probs。
    log_probs: list[float] = field(default_factory=list)
    # rewards: rewards 数据。
    rewards: list[float] = field(default_factory=list)
    # dones: dones 数据。
    dones: list[bool] = field(default_factory=list)
    # values: values 数据。
    values: list[float] = field(default_factory=list)
    # action_masks: action masks used when the action was sampled.
    action_masks: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        action_mask: np.ndarray | None = None,
    ) -> None:
        """处理add 数据相关业务逻辑。

        参数：
            observation: 观测向量，类型为 np.ndarray。
            action: 动作，类型为 int。
            log_prob: 日志prob，类型为 float。
            reward: 奖励，类型为 float。
            done: 结束标记，类型为 bool。
            value: 数值，类型为 float。

        返回：
            无返回值；通过状态变更、文件输出或日志记录体现执行结果。
        """
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        if action_mask is None:
            self.action_masks.append(np.ones(1, dtype=np.float32))
        else:
            self.action_masks.append(np.asarray(action_mask, dtype=np.float32))

    def clear(self) -> None:
        """处理clear 数据相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            无返回值；通过状态变更、文件输出或日志记录体现执行结果。
        """
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.action_masks.clear()

    def __len__(self) -> int:
        """处理len 数据相关业务逻辑。

        参数：
            无显式业务参数。

        返回：
            int，表示该函数计算或构建得到的结果。
        """
        return len(self.actions)

    def compute_returns_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        last_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算指定指标或中间结果，处理returnsadvantages相关数据。

        参数：
            gamma: 折扣因子，类型为 float。
            gae_lambda: gaelambda，类型为 float。
            last_value: last数值，类型为 float。

        返回：
            tuple[np.ndarray, np.ndarray]，表示该函数计算或构建得到的结果。
        """
        values = self.values + [float(last_value)]
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        gae = 0.0
        for index in reversed(range(len(self.rewards))):
            mask = 0.0 if self.dones[index] else 1.0
            delta = self.rewards[index] + gamma * values[index + 1] * mask - values[index]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[index] = gae
        returns = advantages + np.asarray(self.values, dtype=np.float32)
        if len(advantages) > 1 and float(np.std(advantages)) > 1e-8:
            advantages = (advantages - float(np.mean(advantages))) / (
                float(np.std(advantages)) + 1e-8
            )
        return returns.astype(np.float32), advantages.astype(np.float32)
