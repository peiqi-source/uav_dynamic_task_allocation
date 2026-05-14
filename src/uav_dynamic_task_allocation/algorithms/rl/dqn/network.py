"""DQN 算法模块中的神经网络实现。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from uav_dynamic_task_allocation.utils.config import get_config_value


class DQNNetworkError(Exception):
    """DQN 网络结构、输入输出维度或动作 mask 处理过程中的自定义错误。"""


@dataclass(frozen=True)
class DQNNetworkConfig:
    """
    DQN 网络配置。

    该配置描述 Q 网络的基础结构。根据论文中的 DQN 网络思想，
    Q 网络接收当前状态特征 S，输出每个动作 A 对应的 Q 值。

    input_dim:
        observation 向量长度，对应 ObservationBuilder 输出的 vector 维度。

    action_dim:
        固定动作空间大小，对应 FixedUAVTargetActionSpace.size。
        在当前工程中通常等于 max_uavs * max_targets。

    hidden_dims:
        MLP 隐藏层维度。论文中使用前馈神经网络和全连接隐藏层，
        这里默认使用两层 256 维隐藏层，后续可通过 YAML 调整。

    activation:
        隐藏层激活函数。当前支持 relu、tanh、gelu。

    dropout:
        dropout 概率。强化学习中 dropout 不一定稳定，因此默认 0。

    use_layer_norm:
        是否在隐藏层后使用 LayerNorm。它是工程增强项，用于改善特征尺度差异较大时的训练稳定性。
    """

    # input_dim: 输入dim。
    input_dim: int
    # action_dim: 动作dim。
    action_dim: int
    # hidden_dims: hiddendims。
    hidden_dims: tuple[int, ...] = (256, 256)
    # activation: activation 数据。
    activation: str = "relu"
    # dropout: dropout 数据。
    dropout: float = 0.0
    # use_layer_norm: uselayernorm。
    use_layer_norm: bool = False
    # network_type: 神经网络类型。
    network_type: str = "dqn"

    def validate(self) -> None:
        """检查网络配置是否合法，避免模型初始化后才出现维度错误。"""
        if self.input_dim <= 0:
            raise DQNNetworkError("input_dim must be positive.")

        if self.action_dim <= 0:
            raise DQNNetworkError("action_dim must be positive.")

        if not self.hidden_dims:
            raise DQNNetworkError("hidden_dims must not be empty.")

        for hidden_dim in self.hidden_dims:
            if hidden_dim <= 0:
                raise DQNNetworkError(
                    f"All hidden dimensions must be positive, got {hidden_dim}."
                )

        if self.activation not in {"relu", "tanh", "gelu"}:
            raise DQNNetworkError(
                f"Unsupported activation: {self.activation}. "
                "Available options: relu, tanh, gelu."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise DQNNetworkError("dropout must be in [0, 1).")

        if self.network_type not in {"dqn", "attention_dqn"}:
            raise DQNNetworkError(
                f"Unsupported network_type: {self.network_type}. "
                "Available options: dqn, attention_dqn."
            )


class DQNNetwork(nn.Module):
    """
    DQN Q 网络。

    网络输入：
        observation vector，形状为 [batch_size, input_dim]

    网络输出：
        q_values，形状为 [batch_size, action_dim]

    每个输出维度对应一个固定 action_id。
    例如 action_dim=6000 时，网络一次性输出 6000 个动作的 Q 值。
    后续选择动作时，会结合 action mask 屏蔽非法动作。
    """

    def __init__(self, config: DQNNetworkConfig) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            config: 配置对象，类型为 DQNNetworkConfig。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        super().__init__()

        config.validate()
        # config: 配置。
        self.config = config

        layers: list[nn.Module] = []
        previous_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))

            if config.use_layer_norm:
                # LayerNorm 对每个样本的隐藏特征做归一化。
                # 这里保留为可选项，是因为 UAV/Target 坐标、距离、价值等特征尺度差异较大。
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(self._build_activation(config.activation))

            if config.dropout > 0:
                # dropout 在监督学习中常用于防过拟合，但在强化学习中可能影响值函数稳定性。
                # 因此默认关闭，只有实验需要时再通过配置打开。
                layers.append(nn.Dropout(p=config.dropout))

            previous_dim = hidden_dim

        # 输出层不加激活函数。
        # Q 值可以是正数，也可以是负数，不能被 ReLU 等激活函数截断。
        layers.append(nn.Linear(previous_dim, config.action_dim))

        # model: 模型。
        self.model = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            observations:
                形状为 [batch_size, input_dim] 的状态向量。

        Returns:
            形状为 [batch_size, action_dim] 的 Q 值矩阵。
        """
        if observations.ndim != 2:
            raise DQNNetworkError(
                "observations must be a 2D tensor with shape "
                f"[batch_size, input_dim], got {tuple(observations.shape)}"
            )

        if observations.shape[1] != self.config.input_dim:
            raise DQNNetworkError(
                "Observation dimension mismatch: "
                f"expected={self.config.input_dim}, "
                f"got={observations.shape[1]}"
            )

        return self.model(observations)

    def _initialize_weights(self) -> None:
        """
        初始化网络参数。

        对 Linear 层使用 Kaiming 初始化，适合 ReLU 类激活函数。
        bias 初始化为 0，使网络初始输出更稳定。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        """根据配置名称创建激活函数。"""
        if name == "relu":
            return nn.ReLU()

        if name == "tanh":
            return nn.Tanh()

        if name == "gelu":
            return nn.GELU()

        raise DQNNetworkError(f"Unsupported activation: {name}")


def apply_action_mask(
    q_values: torch.Tensor,
    action_mask: torch.Tensor,
    invalid_value: float = -1e9,
) -> torch.Tensor:
    """
    对 Q 值应用 action mask。

    DQN 网络会输出所有固定动作的 Q 值，但当前状态下并不是所有动作都合法。
    action_mask 中：
        1 表示合法动作；
        0 表示非法动作。

    处理方式：
        将非法动作对应的 Q 值替换为一个极小值。
        这样 argmax 时就不会选择非法动作。

    Args:
        q_values:
            形状为 [batch_size, action_dim] 的 Q 值。
        action_mask:
            形状为 [batch_size, action_dim] 或 [action_dim] 的 mask。
        invalid_value:
            非法动作替换值，默认 -1e9。

    Returns:
        masked_q_values，形状与 q_values 相同。
    """
    if q_values.ndim != 2:
        raise DQNNetworkError(
            f"q_values must be 2D, got shape={tuple(q_values.shape)}"
        )

    if action_mask.ndim == 1:
        action_mask = action_mask.unsqueeze(0)

    if action_mask.ndim != 2:
        raise DQNNetworkError(
            f"action_mask must be 1D or 2D, got shape={tuple(action_mask.shape)}"
        )

    if action_mask.shape != q_values.shape:
        raise DQNNetworkError(
            "action_mask shape must match q_values shape, "
            f"got mask={tuple(action_mask.shape)}, q_values={tuple(q_values.shape)}"
        )

    valid_counts = action_mask.sum(dim=1)
    if torch.any(valid_counts <= 0):
        raise DQNNetworkError(
            "At least one sample has no valid action. "
            "Please check action_space configuration or environment state."
        )

    return q_values.masked_fill(action_mask <= 0, invalid_value)


def select_greedy_action(
    q_values: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    根据 Q 值选择贪心动作。

    如果提供 action_mask，则只在合法动作中选择 Q 值最大的动作。
    如果不提供 action_mask，则直接对所有动作取 argmax。

    Returns:
        action_ids，形状为 [batch_size]。
    """
    if action_mask is not None:
        q_values = apply_action_mask(q_values, action_mask)

    return torch.argmax(q_values, dim=1)


def load_dqn_network_config(
    config: dict[str, Any],
    input_dim: int,
    action_dim: int,
) -> DQNNetworkConfig:
    """
    从项目总配置中读取 DQN 网络配置。

    input_dim 和 action_dim 不直接写死在 YAML 中，而是由当前 observation 配置
    和 action_space 配置计算得到。这样可以避免配置维度和真实环境维度不一致。
    """
    hidden_dims_raw = get_config_value(
        config,
        "dqn.network.hidden_dims",
        default=[256, 256],
    )

    hidden_dims = tuple(int(value) for value in hidden_dims_raw)

    network_config = DQNNetworkConfig(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activation=str(
            get_config_value(config, "dqn.network.activation", default="relu")
        ),
        dropout=float(
            get_config_value(config, "dqn.network.dropout", default=0.0)
        ),
        use_layer_norm=bool(
            get_config_value(
                config,
                "dqn.network.use_layer_norm",
                default=False,
            )
        ),
        network_type=str(
            get_config_value(config, "dqn.network.network_type", default="dqn")
        ),
    )

    network_config.validate()
    return network_config
