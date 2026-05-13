"""PPO 算法模块中的训练器实现。"""
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

    # max_episodes: 最大值episodes。
    max_episodes: int = 20
    # seed: 随机种子。
    seed: int = 42
    # device: 计算设备。
    device: str = "cpu"


class PPOClustererTrainer:
    """Small PPO trainer for TargetGroupingEnv."""

    def __init__(
        self,
        env: TargetGroupingEnv,
        agent_config: PPOAgentConfig,
        trainer_config: PPOTrainerConfig,
    ) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            env: 环境对象，类型为 TargetGroupingEnv。
            agent_config: 智能体训练配置，类型为 PPOAgentConfig。
            trainer_config: trainer_config 参数，类型为 PPOTrainerConfig。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        if torch is None:
            raise PPOAgentError("PyTorch is required for PPOClustererTrainer.")
        # env: 环境。
        self.env = env
        # agent_config: 智能体配置。
        self.agent_config = agent_config
        # trainer_config: 训练器配置。
        self.trainer_config = trainer_config
        observation = env.reset()
        # action_dim: 动作dim。
        self.action_dim = env.action_dim
        # agent: 智能体。
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
        """执行模型训练流程并保存训练产物。

        参数：
            无显式业务参数。

        返回：
            tuple[list[dict[str, float]], PPOAgent]，表示该函数计算或构建得到的结果。
        """
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
