"""DQN 算法模块中的检查点实现。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from uav_dynamic_task_allocation.algorithms.rl.dqn.agent import DQNAgent
from uav_dynamic_task_allocation.utils.config import get_config_value, resolve_path


class DQNCheckpointError(Exception):
    """DQN checkpoint 保存、加载和校验过程中的自定义错误。"""


@dataclass(frozen=True)
class DQNCheckpointConfig:
    """
    DQN checkpoint 配置。

    checkpoint_dir:
        模型保存目录。

    latest_filename:
        最近一次模型文件名，通常用于断点续训。

    best_filename:
        当前最优模型文件名，通常用于最终评估。

    save_optimizer:
        是否保存 optimizer 状态。如果后续要继续训练，应该设为 True。

    save_replay_buffer:
        是否保存 replay buffer。当前项目先不启用，因为 replay buffer 可能较大，
        而且我们还没有实现 buffer 的序列化恢复逻辑。
    """

    # enabled: enabled 数据。
    enabled: bool = True
    # checkpoint_dir: 检查点dir。
    checkpoint_dir: str = "checkpoints/dqn"
    # latest_filename: latestfilename。
    latest_filename: str = "latest.pt"
    # best_filename: bestfilename。
    best_filename: str = "best.pt"
    # save_optimizer: saveoptimizer。
    save_optimizer: bool = True
    # save_replay_buffer: savereplay经验缓冲区。
    save_replay_buffer: bool = False

    def validate(self) -> None:
        """检查 checkpoint 配置是否合法。"""
        if not self.checkpoint_dir:
            raise DQNCheckpointError("checkpoint_dir must not be empty.")

        if not self.latest_filename.endswith(".pt"):
            raise DQNCheckpointError("latest_filename should end with .pt.")

        if not self.best_filename.endswith(".pt"):
            raise DQNCheckpointError("best_filename should end with .pt.")

        if self.save_replay_buffer:
            raise DQNCheckpointError(
                "save_replay_buffer=True is not supported yet. "
                "Replay buffer serialization will be implemented later."
            )


@dataclass
class DQNCheckpointMetadata:
    """
    checkpoint 元信息。

    这部分不直接参与模型计算，但对实验管理非常重要。
    例如你以后看到一个 best.pt，需要知道它来自第几个 episode、
    当时 reward 是多少、网络更新了多少次、epsilon 下降到了多少。
    """

    # episode: 训练回合。
    episode: int = 0
    # global_env_steps: global环境步数。
    global_env_steps: int = 0

    # total_action_steps: total动作步数。
    total_action_steps: int = 0
    # total_update_steps: totalupdate步数。
    total_update_steps: int = 0

    # best_reward: best奖励。
    best_reward: float | None = None
    # last_reward: last奖励。
    last_reward: float | None = None
    # epsilon: 探索率。
    epsilon: float | None = None

    # created_at: createdat。
    created_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # extra: extra 数据。
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DQNCheckpointLoadResult:
    """
    checkpoint 加载结果。

    返回 metadata 和原始 checkpoint 字典，便于后续恢复训练器状态或调试。
    """

    # path: 路径。
    path: Path
    # metadata: 扩展元数据。
    metadata: DQNCheckpointMetadata
    # raw_checkpoint: raw检查点。
    raw_checkpoint: dict[str, Any]


class DQNCheckpointManager:
    """
    DQN checkpoint 管理器。

    它负责：
    1. 保存 policy network；
    2. 保存 target network；
    3. 保存 optimizer 状态；
    4. 保存 agent 的训练步数；
    5. 保存 episode、reward、epsilon 等元信息；
    6. 从 checkpoint 恢复 agent 状态。

    为什么不直接在 trainer.py 里写 torch.save？
    因为保存和加载模型是一套独立逻辑，后面 train_dqn.py、evaluate_dqn.py、
    resume training 都会用到。如果写死在 trainer.py 里，会越来越难维护。
    """

    def __init__(
        self,
        config: DQNCheckpointConfig,
        project_root: str | Path | None = None,
    ) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            config: 配置对象，类型为 DQNCheckpointConfig。
            project_root: project_root 参数，类型为 str | Path | None。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        config.validate()

        # config: 配置。
        self.config = config
        # project_root: projectroot。
        self.project_root = Path(project_root) if project_root is not None else None

        # checkpoint_dir: 检查点dir。
        self.checkpoint_dir = resolve_path(
            self.config.checkpoint_dir,
            project_root=self.project_root,
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        """返回 latest checkpoint 路径。"""
        return self.checkpoint_dir / self.config.latest_filename

    @property
    def best_path(self) -> Path:
        """返回 best checkpoint 路径。"""
        return self.checkpoint_dir / self.config.best_filename

    def save(
        self,
        agent: DQNAgent,
        metadata: DQNCheckpointMetadata,
        path: str | Path | None = None,
        is_best: bool = False,
    ) -> Path:
        """
        保存 DQNAgent 状态。

        Args:
            agent:
                当前 DQN 智能体。
            metadata:
                当前训练元信息。
            path:
                自定义保存路径。如果不传，则默认保存到 latest.pt 或 best.pt。
            is_best:
                是否保存为 best checkpoint。

        Returns:
            实际保存路径。
        """
        if not self.config.enabled:
            raise DQNCheckpointError(
                "Checkpoint is disabled in DQNCheckpointConfig."
            )

        save_path = self._resolve_save_path(path=path, is_best=is_best)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "metadata": asdict(metadata),
            "network_config": asdict(agent.network_config),
            "agent_config": asdict(agent.agent_config),
            "policy_network_state_dict": agent.policy_network.state_dict(),
            "target_network_state_dict": agent.target_network.state_dict(),
            "total_action_steps": agent.total_action_steps,
            "total_update_steps": agent.total_update_steps,
        }

        if self.config.save_optimizer:
            checkpoint["optimizer_state_dict"] = agent.optimizer.state_dict()

        torch.save(checkpoint, save_path)

        return save_path

    def load(
        self,
        agent: DQNAgent,
        path: str | Path,
        load_optimizer: bool = True,
        map_location: str | torch.device | None = None,
    ) -> DQNCheckpointLoadResult:
        """
        从 checkpoint 恢复 DQNAgent 状态。

        Args:
            agent:
                要恢复参数的 DQNAgent。
            path:
                checkpoint 文件路径。
            load_optimizer:
                是否恢复 optimizer 状态。继续训练时建议 True，评估时可以 False。
            map_location:
                torch.load 的设备映射参数。比如从 GPU 模型加载到 CPU 时很有用。

        Returns:
            DQNCheckpointLoadResult。
        """
        checkpoint_path = resolve_path(path, project_root=self.project_root)

        if not checkpoint_path.exists():
            raise DQNCheckpointError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location or agent.device,
        )

        self._validate_checkpoint_dict(checkpoint)

        agent.policy_network.load_state_dict(
            checkpoint["policy_network_state_dict"]
        )
        agent.target_network.load_state_dict(
            checkpoint["target_network_state_dict"]
        )

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        agent.total_action_steps = int(checkpoint.get("total_action_steps", 0))
        agent.total_update_steps = int(checkpoint.get("total_update_steps", 0))

        metadata = DQNCheckpointMetadata(**checkpoint["metadata"])

        return DQNCheckpointLoadResult(
            path=checkpoint_path,
            metadata=metadata,
            raw_checkpoint=checkpoint,
        )

    def save_latest(
        self,
        agent: DQNAgent,
        metadata: DQNCheckpointMetadata,
    ) -> Path:
        """保存 latest checkpoint。"""
        return self.save(
            agent=agent,
            metadata=metadata,
            path=self.latest_path,
            is_best=False,
        )

    def save_best(
        self,
        agent: DQNAgent,
        metadata: DQNCheckpointMetadata,
    ) -> Path:
        """保存 best checkpoint。"""
        return self.save(
            agent=agent,
            metadata=metadata,
            path=self.best_path,
            is_best=True,
        )

    def load_latest(
        self,
        agent: DQNAgent,
        load_optimizer: bool = True,
        map_location: str | torch.device | None = None,
    ) -> DQNCheckpointLoadResult:
        """加载 latest checkpoint。"""
        return self.load(
            agent=agent,
            path=self.latest_path,
            load_optimizer=load_optimizer,
            map_location=map_location,
        )

    def load_best(
        self,
        agent: DQNAgent,
        load_optimizer: bool = False,
        map_location: str | torch.device | None = None,
    ) -> DQNCheckpointLoadResult:
        """加载 best checkpoint。默认不加载 optimizer，因为 best 通常用于评估。"""
        return self.load(
            agent=agent,
            path=self.best_path,
            load_optimizer=load_optimizer,
            map_location=map_location,
        )

    def _resolve_save_path(
        self,
        path: str | Path | None,
        is_best: bool,
    ) -> Path:
        """
        解析保存路径。

        优先级：
        1. 如果用户显式传入 path，就使用 path；
        2. 如果 is_best=True，使用 best_path；
        3. 否则使用 latest_path。
        """
        if path is not None:
            return resolve_path(path, project_root=self.project_root)

        if is_best:
            return self.best_path

        return self.latest_path

    @staticmethod
    def _validate_checkpoint_dict(checkpoint: dict[str, Any]) -> None:
        """
        检查 checkpoint 文件是否包含必要字段。

        这个检查可以避免加载了错误文件，比如误把 metrics.csv 或其他 .pt 文件当作 DQN 模型。
        """
        required_keys = [
            "metadata",
            "network_config",
            "agent_config",
            "policy_network_state_dict",
            "target_network_state_dict",
        ]

        missing_keys = [
            key for key in required_keys
            if key not in checkpoint
        ]

        if missing_keys:
            raise DQNCheckpointError(
                f"Invalid checkpoint. Missing keys: {missing_keys}"
            )


def load_dqn_checkpoint_config(config: dict[str, Any]) -> DQNCheckpointConfig:
    """
    从项目总配置中读取 DQN checkpoint 配置。

    这样 checkpoint 保存路径、文件名、是否保存 optimizer 都由 YAML 控制，
    而不是写死在训练脚本里。
    """
    checkpoint_config = DQNCheckpointConfig(
        enabled=bool(
            get_config_value(
                config,
                "dqn.checkpoint.enabled",
                default=True,
            )
        ),
        checkpoint_dir=str(
            get_config_value(
                config,
                "dqn.checkpoint.checkpoint_dir",
                default="checkpoints/dqn",
            )
        ),
        latest_filename=str(
            get_config_value(
                config,
                "dqn.checkpoint.latest_filename",
                default="latest.pt",
            )
        ),
        best_filename=str(
            get_config_value(
                config,
                "dqn.checkpoint.best_filename",
                default="best.pt",
            )
        ),
        save_optimizer=bool(
            get_config_value(
                config,
                "dqn.checkpoint.save_optimizer",
                default=True,
            )
        ),
        save_replay_buffer=bool(
            get_config_value(
                config,
                "dqn.checkpoint.save_replay_buffer",
                default=False,
            )
        ),
    )

    checkpoint_config.validate()
    return checkpoint_config