from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_dynamic_task_allocation.algorithms.rl.ppo.agent import (
    PPOAgent,
    PPOAgentConfig,
    PPOAgentError,
)
from uav_dynamic_task_allocation.algorithms.rl.ppo.buffer import PPORolloutBuffer
from uav_dynamic_task_allocation.algorithms.rl.ppo.network import PPONetworkConfig, torch
from uav_dynamic_task_allocation.envs.target_grouping_env import TargetGroupingEnv


@dataclass(frozen=True)
class PPOTrainerConfig:
    """Training loop configuration for target regrouping PPO."""

    max_episodes: int = 20
    seed: int = 42
    device: str = "cpu"


class PPOClustererTrainer:
    """Small PPO trainer for TargetGroupingEnv."""

    def __init__(
        self,
        env: TargetGroupingEnv,
        agent_config: PPOAgentConfig,
        trainer_config: PPOTrainerConfig,
    ) -> None:
        if torch is None:
            raise PPOAgentError("PyTorch is required for PPOClustererTrainer.")
        self.env = env
        self.agent_config = agent_config
        self.trainer_config = trainer_config
        observation = env.reset()
        self.action_dim = env.action_dim
        self.agent = PPOAgent(
            network_config=PPONetworkConfig(
                observation_dim=len(observation),
                action_dim=self.action_dim,
            ),
            agent_config=agent_config,
            seed=trainer_config.seed,
            device=trainer_config.device,
        )

    def train(self) -> tuple[list[dict[str, float]], PPOAgent]:
        rng = np.random.default_rng(self.trainer_config.seed)
        rows: list[dict[str, float]] = []
        for episode in range(1, self.trainer_config.max_episodes + 1):
            observation = self.env.reset()
            buffer = PPORolloutBuffer()
            total_reward = 0.0
            done = False
            while not done:
                mask = self.env.action_mask()
                if not mask.any():
                    action = int(rng.integers(0, self.action_dim))
                    log_prob = 0.0
                    value = 0.0
                else:
                    action, log_prob, value = self.agent.select_action(observation, mask)
                next_observation, reward, done, _ = self.env.step_index(action)
                buffer.add(observation, action, log_prob, reward, done, value)
                total_reward += reward
                observation = next_observation
            update_metrics = self.agent.update(buffer)
            rows.append(
                {
                    "episode": float(episode),
                    "total_reward": float(total_reward),
                    **update_metrics,
                }
            )
        return rows, self.agent
