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

    observation_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (128, 128)


if nn is not None:

    class PPOActorCritic(nn.Module):
        """Shared actor-critic MLP for discrete PPO actions."""

        def __init__(self, config: PPONetworkConfig) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            previous_dim = config.observation_dim
            for hidden_dim in config.hidden_dims:
                layers.append(nn.Linear(previous_dim, hidden_dim))
                layers.append(nn.Tanh())
                previous_dim = hidden_dim
            self.encoder = nn.Sequential(*layers)
            self.actor = nn.Linear(previous_dim, config.action_dim)
            self.critic = nn.Linear(previous_dim, 1)

        def forward(self, observations):
            encoded = self.encoder(observations)
            return self.actor(encoded), self.critic(encoded).squeeze(-1)

else:

    class PPOActorCritic:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise PPONetworkUnavailableError("PyTorch is required for PPOActorCritic.")
