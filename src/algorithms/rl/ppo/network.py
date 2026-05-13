"""PPO 算法模块中的神经网络实现。"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


class PPONetworkUnavailableError(Exception):
    """Raised when PPO networks are requested without PyTorch."""


@dataclass(frozen=True)
class PPONetworkConfig:
    """Actor-critic network configuration."""

    # observation_dim: 观测向量dim。
    observation_dim: int
    # action_dim: 动作dim。
    action_dim: int
    # hidden_dims: hiddendims。
    hidden_dims: tuple[int, ...] = (128, 128)


if nn is not None:

    class PPOActorCritic(nn.Module):
        """Shared actor-critic MLP for discrete PPO actions."""

        def __init__(self, config: PPONetworkConfig) -> None:
            """初始化对象并保存运行所需的配置、依赖和内部状态。

            参数：
                config: 配置对象，类型为 PPONetworkConfig。

            返回：
                无返回值；初始化实例属性并完成对象准备。
            """
            super().__init__()
            layers: list[nn.Module] = []
            previous_dim = config.observation_dim
            for hidden_dim in config.hidden_dims:
                layers.append(nn.Linear(previous_dim, hidden_dim))
                layers.append(nn.Tanh())
                previous_dim = hidden_dim
            # encoder: encoder 数据。
            self.encoder = nn.Sequential(*layers)
            # actor: actor 数据。
            self.actor = nn.Linear(previous_dim, config.action_dim)
            # critic: critic 数据。
            self.critic = nn.Linear(previous_dim, 1)

        def forward(self, observations):
            """处理forward 数据相关业务逻辑。

            参数：
                observations: observations 数据。

            返回：
                函数执行结果；具体类型由调用上下文或下游流程决定。
            """
            encoded = self.encoder(observations)
            return self.actor(encoded), self.critic(encoded).squeeze(-1)

else:

    class PPOActorCritic:  # type: ignore[no-redef]
        """PPOActorCritic 类，封装PPO 算法actorcritic相关的数据结构与业务行为。"""
        def __init__(self, *args, **kwargs):
            """初始化对象并保存运行所需的配置、依赖和内部状态。

            参数：
                无显式业务参数。

            返回：
                无返回值；初始化实例属性并完成对象准备。
            """
            raise PPONetworkUnavailableError("PyTorch is required for PPOActorCritic.")
