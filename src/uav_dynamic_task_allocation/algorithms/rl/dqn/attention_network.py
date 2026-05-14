"""DQN 算法模块中的attention神经网络实现。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


class AttentionDQNUnavailableError(Exception):
    """Raised when Attention-DQN is requested without PyTorch installed."""


@dataclass(frozen=True)
class AttentionDQNNetworkConfig:
    """Network configuration for Attention-DQN strike-order planning."""

    # input_dim: 输入dim。
    input_dim: int
    # action_dim: 动作dim。
    action_dim: int
    # hidden_dims: hiddendims。
    hidden_dims: tuple[int, ...] = (256, 256)
    # attention_hidden_dim: attentionhiddendim。
    attention_hidden_dim: int = 128
    # dropout: dropout 数据。
    dropout: float = 0.0
    # feature_priority_indices: featurepriorityindices。
    feature_priority_indices: tuple[int, ...] = ()

    def validate(self) -> None:
        """校验当前对象或输入配置的合法性。

        参数：
            无显式业务参数。

        返回：
            无返回值；通过状态变更、文件输出或日志记录体现执行结果。
        """
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must not be empty.")
        if self.attention_hidden_dim <= 0:
            raise ValueError("attention_hidden_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        for index in self.feature_priority_indices:
            if index < 0 or index >= self.input_dim:
                raise ValueError(f"feature priority index out of range: {index}")


if nn is not None:

    class AttentionDQNNetwork(nn.Module):
        """
        DQN network with feature attention for strike-order decisions.

        The attention gate learns input-feature weights before the Q head. It
        can emphasize target value, defense, distance and residual-resource
        features while remaining compatible with the existing flat
        StrikeOrderEnv observation vector.
        """

        def __init__(
            self,
            config: AttentionDQNNetworkConfig | object | None = None,
            input_dim: int | None = None,
            action_dim: int | None = None,
            hidden_dim: int = 256,
        ) -> None:
            """初始化对象并保存运行所需的配置、依赖和内部状态。

            参数：
                config: 配置对象，类型为 AttentionDQNNetworkConfig | object | None。
                input_dim: input_dim 参数，类型为 int | None。
                action_dim: action_dim 参数，类型为 int | None。
                hidden_dim: hidden_dim 参数，类型为 int。

            返回：
                无返回值；初始化实例属性并完成对象准备。
            """
            super().__init__()
            if config is None:
                if input_dim is None or action_dim is None:
                    raise ValueError("input_dim and action_dim are required.")
                config = AttentionDQNNetworkConfig(
                    input_dim=input_dim,
                    action_dim=action_dim,
                    hidden_dims=(hidden_dim, hidden_dim),
                )
            elif not isinstance(config, AttentionDQNNetworkConfig):
                config = AttentionDQNNetworkConfig(
                    input_dim=int(getattr(config, "input_dim")),
                    action_dim=int(getattr(config, "action_dim")),
                    hidden_dims=tuple(getattr(config, "hidden_dims", (hidden_dim, hidden_dim))),
                    dropout=float(getattr(config, "dropout", 0.0)),
                )
            config.validate()
            # config: 配置。
            self.config = config
            # attention: attention 数据。
            self.attention = nn.Sequential(
                nn.Linear(config.input_dim, config.attention_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.attention_hidden_dim, config.input_dim),
                nn.Sigmoid(),
            )

            layers: list[nn.Module] = []
            previous_dim = config.input_dim
            for hidden in config.hidden_dims:
                layers.append(nn.Linear(previous_dim, int(hidden)))
                layers.append(nn.ReLU())
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
                previous_dim = int(hidden)
            layers.append(nn.Linear(previous_dim, config.action_dim))
            # q_head: qhead。
            self.q_head = nn.Sequential(*layers)

        def forward(self, observation, action_mask=None):
            """处理forward 数据相关业务逻辑。

            参数：
                observation: 观测向量。
                action_mask: 动作掩码。

            返回：
                函数执行结果；具体类型由调用上下文或下游流程决定。
            """
            weights = self.compute_attention_weights(observation)
            q_values = self.q_head(observation * weights)
            if action_mask is not None:
                mask = action_mask.bool()
                q_values = q_values.masked_fill(~mask, -1e9)
            return q_values

        def compute_attention_weights(self, observation):
            """计算指定指标或中间结果，处理attention权重集合相关数据。

            参数：
                observation: 观测向量。

            返回：
                函数执行结果；具体类型由调用上下文或下游流程决定。
            """
            return self.attention(observation)

else:

    class AttentionDQNNetwork:  # type: ignore[no-redef]
        """Placeholder that explains the missing optional dependency."""

        def __init__(self, *args, **kwargs):
            """初始化对象并保存运行所需的配置、依赖和内部状态。

            参数：
                无显式业务参数。

            返回：
                无返回值；初始化实例属性并完成对象准备。
            """
            raise AttentionDQNUnavailableError(
                "PyTorch is required to instantiate AttentionDQNNetwork."
            )


def infer_priority_indices(
    input_dim: int,
    *,
    target_value_indices: Sequence[int] = (),
    defense_indices: Sequence[int] = (),
    distance_indices: Sequence[int] = (),
    resource_indices: Sequence[int] = (),
) -> tuple[int, ...]:
    """Build a validated tuple of feature indices used for attention analysis."""
    candidates = (
        list(target_value_indices)
        + list(defense_indices)
        + list(distance_indices)
        + list(resource_indices)
    )
    return tuple(sorted({int(index) for index in candidates if 0 <= int(index) < input_dim}))
