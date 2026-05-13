from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PPORolloutBuffer:
    """Rollout buffer for clipped-objective PPO updates."""

    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.actions)

    def compute_returns_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        last_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
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
