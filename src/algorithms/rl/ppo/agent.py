from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_dynamic_task_allocation.algorithms.rl.ppo.buffer import PPORolloutBuffer
from uav_dynamic_task_allocation.algorithms.rl.ppo.network import (
    PPOActorCritic,
    PPONetworkConfig,
    torch,
)


class PPOAgentError(Exception):
    """Raised when PPO agent operations fail."""


@dataclass(frozen=True)
class PPOAgentConfig:
    """PPO optimization hyperparameters."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    batch_size: int = 64
    update_epochs: int = 4


class PPOAgent:
    """Discrete-action PPO agent with clipped objective."""

    def __init__(
        self,
        network_config: PPONetworkConfig,
        agent_config: PPOAgentConfig,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        if torch is None:
            raise PPOAgentError("PyTorch is required for PPOAgent.")
        torch.manual_seed(seed)
        self.config = agent_config
        self.device = torch.device(device)
        self.network = PPOActorCritic(network_config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=agent_config.learning_rate,
        )

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        logits, value = self.network(obs)
        logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def update(self, buffer: PPORolloutBuffer) -> dict[str, float]:
        if len(buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        returns, advantages = buffer.compute_returns_advantages(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        observations = torch.as_tensor(np.asarray(buffer.observations), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        num_samples = len(buffer)
        batch_size = min(self.config.batch_size, num_samples)
        last_policy_loss = last_value_loss = last_entropy = 0.0

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, batch_size):
                indices = permutation[start:start + batch_size]
                logits, values = self.network(observations[indices])
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions[indices])
                ratios = torch.exp(log_probs - old_log_probs[indices])
                unclipped = ratios * advantages_t[indices]
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advantages_t[indices]
                policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
                value_loss = torch.mean((returns_t[indices] - values) ** 2)
                entropy = torch.mean(dist.entropy())
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy.item())

        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
        }
