"""planning 数据模块中的打击顺序规划器实现。"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from uav_dynamic_task_allocation.core.contracts import (
    AlgorithmMetadata,
    AllocationPlan,
    PipelineStage,
    StrikeOrderPlan,
    TargetCluster,
    TargetClusterSet,
)
from uav_dynamic_task_allocation.core.entities import Position
from uav_dynamic_task_allocation.envs.strike_order_env import (
    StrikeOrderEnv,
    StrikeOrderEnvConfig,
    load_strike_order_env_config,
)
from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    get_project_root,
    resolve_path,
)
from uav_dynamic_task_allocation.utils.device import get_device


class StrikeOrderPlannerError(Exception):
    """群内打击次序规划过程中的自定义错误。"""


@dataclass(frozen=True)
class StrikeOrderPlannerConfig:
    """
    群内打击次序规划配置。

    method:
        dqn / nearest_neighbor / random

    start_position_source:
        cluster_center:
            默认方式。表示 DQN 排序阶段假设 UAV 小组已经进入目标群任务区。
        assignment_real_time_position:
            使用资源分配结果中的实时位置。
        assignment_start_position:
            使用资源分配结果中的起飞位置。
    """

    # method: method 数据。
    method: str = "dqn"
    # checkpoint_type: 检查点类型。
    checkpoint_type: str = "best"

    # allow_fallback: allowfallback。
    allow_fallback: bool = True
    # fallback_method: fallbackmethod。
    fallback_method: str = "nearest_neighbor"

    # start_position_source: start位置坐标source。
    start_position_source: str = "cluster_center"

    # device: 计算设备。
    device: str = "auto"
    # random_seed: 随机随机种子。
    random_seed: int = 42

    # debug_csv_path: 调试 CSV 输出路径。
    debug_csv_path: str = "outputs/intermediate/strike_order_planner.csv"

    def validate(self) -> None:
        """检查配置是否合法。"""
        supported_methods = {"dqn", "attention_dqn", "nearest_neighbor", "random"}

        if self.method not in supported_methods:
            raise StrikeOrderPlannerError(
                f"Unsupported strike order method={self.method}. "
                f"Supported methods: {sorted(supported_methods)}"
            )

        if self.fallback_method not in {"nearest_neighbor", "random"}:
            raise StrikeOrderPlannerError(
                "fallback_method must be nearest_neighbor or random."
            )

        if self.checkpoint_type not in {"best", "latest"}:
            raise StrikeOrderPlannerError("checkpoint_type must be best or latest.")

        if self.start_position_source not in {
            "cluster_center",
            "assignment_real_time_position",
            "assignment_start_position",
        }:
            raise StrikeOrderPlannerError(
                "start_position_source must be cluster_center, "
                "assignment_real_time_position, or assignment_start_position."
            )


@dataclass
class StrikeOrderPlannerRecord:
    """单个目标群的打击次序规划记录。"""

    # cluster_id: 目标簇编号。
    cluster_id: int
    # method: method 数据。
    method: str
    # ordered_target_ids: ordered目标编号集合。
    ordered_target_ids: list[int]
    # total_path_distance: total路径distance。
    total_path_distance: float
    # expected_reward: expected奖励。
    expected_reward: float
    # start_position: start位置坐标。
    start_position: Position
    # path_positions: 路径positions。
    path_positions: list[Position]
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrikeOrderPlannerResult:
    """
    StrikeOrderPlanner 输出结果。

    strike_order_plans:
        标准合同结构，key 是 cluster_id，value 是 StrikeOrderPlan。

    records:
        用于 debug CSV 和报告分析。
    """

    # strike_order_plans: 打击顺序plans。
    strike_order_plans: dict[int, StrikeOrderPlan]
    # records: records 数据。
    records: list[StrikeOrderPlannerRecord]
    # algorithm_metadata: algorithm扩展元数据。
    algorithm_metadata: AlgorithmMetadata
    # metadata: 扩展元数据。
    metadata: dict[str, Any] = field(default_factory=dict)


class StrikeOrderPlanner:
    """
    群内打击次序规划器。

    输入：
        TargetClusterSet
        AllocationPlan

    输出：
        dict[int, StrikeOrderPlan]

    当前支持：
        dqn
        nearest_neighbor
        random

    设计上，StrikeOrderPlanner 不关心目标是怎么筛选、怎么分群、资源是怎么分配的。
    它只负责给每个 TargetCluster 生成群内打击顺序。
    """

    def __init__(
        self,
        planner_config: StrikeOrderPlannerConfig,
        env_config: StrikeOrderEnvConfig,
        full_config: dict[str, Any],
        logger: logging.Logger | None = None,
    ) -> None:
        """初始化对象并保存运行所需的配置、依赖和内部状态。

        参数：
            planner_config: 任务规划器配置，类型为 StrikeOrderPlannerConfig。
            env_config: env_config 参数，类型为 StrikeOrderEnvConfig。
            full_config: full_config 参数，类型为 dict[str, Any]。
            logger: 日志器，类型为 logging.Logger | None。

        返回：
            无返回值；初始化实例属性并完成对象准备。
        """
        planner_config.validate()

        # config: 配置。
        self.config = planner_config
        # env_config: 环境配置。
        self.env_config = env_config
        # full_config: full配置。
        self.full_config = full_config
        # logger: 日志器。
        self.logger = logger or logging.getLogger(__name__)

        # rng: rng 数据。
        self.rng = np.random.default_rng(self.config.random_seed)
        # _dqn_agent: DQN 算法智能体。
        self._dqn_agent: DQNAgent | None = None
        # _dqn_load_error: DQN 算法loaderror。
        self._dqn_load_error: Exception | None = None

    def plan(
        self,
        target_clusters: TargetClusterSet,
        allocation_plan: AllocationPlan,
    ) -> StrikeOrderPlannerResult:
        """
        为所有目标群生成打击顺序。
        """
        target_clusters.validate()
        allocation_plan.validate()

        method_used = self.config.method

        if method_used in {"dqn", "attention_dqn"}:
            try:
                self._ensure_dqn_agent(target_clusters)
            except Exception as exc:
                self._dqn_load_error = exc

                if not self.config.allow_fallback:
                    raise StrikeOrderPlannerError(
                        "Failed to load DQN strike order model, "
                        "and allow_fallback=False."
                    ) from exc

                method_used = self.config.fallback_method
                self.logger.warning(
                    "Failed to load strike-order model. "
                    f"requested_method={self.config.method}, "
                    f"fallback_method={method_used}, "
                    f"error={exc}"
                )

        plans: dict[int, StrikeOrderPlan] = {}
        records: list[StrikeOrderPlannerRecord] = []

        for cluster in target_clusters.clusters:
            start_position = self._resolve_start_position(
                cluster=cluster,
                allocation_plan=allocation_plan,
            )

            plan, record = self._plan_single_cluster(
                cluster=cluster,
                start_position=start_position,
                method=method_used,
            )

            plans[cluster.cluster_id] = plan
            records.append(record)

        algorithm_metadata = AlgorithmMetadata(
            algorithm_name="strike_order_planner",
            algorithm_type=method_used,
            stage=PipelineStage.STRIKE_ORDER_PLANNING,
            version="v1",
            config={
                "method": self.config.method,
                "method_used": method_used,
                "checkpoint_type": self.config.checkpoint_type,
                "start_position_source": self.config.start_position_source,
                "allow_fallback": self.config.allow_fallback,
                "fallback_method": self.config.fallback_method,
            },
            notes=(
                "Cluster-level strike-order planning. "
                "It generates ordered target ids for each TargetCluster."
            ),
        )

        return StrikeOrderPlannerResult(
            strike_order_plans=plans,
            records=records,
            algorithm_metadata=algorithm_metadata,
            metadata={
                "requested_method": self.config.method,
                "method_used": method_used,
                "num_clusters": len(target_clusters.clusters),
                "dqn_load_error": str(self._dqn_load_error)
                if self._dqn_load_error is not None
                else "",
            },
        )

    def write_debug_csv(
        self,
        result: StrikeOrderPlannerResult,
        output_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> Path:
        """保存群内打击次序规划结果。"""
        path = resolve_path(
            output_path or self.config.debug_csv_path,
            project_root=project_root,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "cluster_id",
            "method",
            "ordered_target_ids",
            "total_path_distance",
            "expected_reward",
            "start_x",
            "start_y",
            "path_positions",
            "metadata",
        ]

        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for record in result.records:
                writer.writerow(
                    {
                        "cluster_id": record.cluster_id,
                        "method": record.method,
                        "ordered_target_ids": record.ordered_target_ids,
                        "total_path_distance": record.total_path_distance,
                        "expected_reward": record.expected_reward,
                        "start_x": record.start_position.x,
                        "start_y": record.start_position.y,
                        "path_positions": [
                            [position.x, position.y]
                            for position in record.path_positions
                        ],
                        "metadata": record.metadata,
                    }
                )

        return path

    def _plan_single_cluster(
        self,
        cluster: TargetCluster,
        start_position: Position,
        method: str,
    ) -> tuple[StrikeOrderPlan, StrikeOrderPlannerRecord]:
        """为单个目标群生成打击顺序。"""
        env = StrikeOrderEnv(
            target_cluster=cluster,
            start_position=start_position,
            config=self.env_config,
        )

        observation = env.reset()

        while True:
            if int(observation.action_mask.sum()) == 0:
                break

            if method in {"dqn", "attention_dqn"}:
                action_id = self._select_dqn_action(
                    observation_vector=observation.vector,
                    action_mask=observation.action_mask,
                )
            elif method == "nearest_neighbor":
                action_id = self._select_nearest_action(env)
            elif method == "random":
                action_id = self._select_random_action(env)
            else:
                raise StrikeOrderPlannerError(
                    f"Unsupported strike order method: {method}"
                )

            step_result = env.step(action_id)
            observation = step_result.observation

            if step_result.done:
                break

        plan_dict = env.get_strike_order_plan_dict()

        strike_plan = StrikeOrderPlan(
            cluster_id=cluster.cluster_id,
            ordered_target_ids=plan_dict["ordered_target_ids"],
            path_positions=list(env.path_positions),
            total_path_distance=float(plan_dict["total_path_distance"]),
            expected_reward=float(plan_dict["episode_reward"]),
            repeated_target_count=0,
            algorithm_metadata=AlgorithmMetadata(
                algorithm_name=f"strike_order_{method}",
                algorithm_type=method,
                stage=PipelineStage.STRIKE_ORDER_PLANNING,
                version="v1",
                config={
                    "cluster_id": cluster.cluster_id,
                    "num_targets": cluster.num_targets,
                },
            ),
            metadata={
                "method": method,
                "num_targets": cluster.num_targets,
                "start_position": [start_position.x, start_position.y],
            },
        )
        strike_plan.validate()

        record = StrikeOrderPlannerRecord(
            cluster_id=cluster.cluster_id,
            method=method,
            ordered_target_ids=list(strike_plan.ordered_target_ids),
            total_path_distance=float(strike_plan.total_path_distance or 0.0),
            expected_reward=float(strike_plan.expected_reward or 0.0),
            start_position=start_position,
            path_positions=list(strike_plan.path_positions),
            metadata={
                "num_targets": cluster.num_targets,
                "target_ids": cluster.target_ids,
            },
        )

        return strike_plan, record

    def _resolve_start_position(
        self,
        cluster: TargetCluster,
        allocation_plan: AllocationPlan,
    ) -> Position:
        """
        解析群内打击排序的起点。

        默认使用 cluster.center，因为 DQN 排序阶段假设 UAV 小组已经进入目标群任务区。
        后续 MissionSimulator 时间推进后，可以改用 real_time_position。
        """
        if self.config.start_position_source == "cluster_center":
            return cluster.center

        assignment = allocation_plan.get_assignment_by_cluster_id(
            cluster.cluster_id
        )

        if self.config.start_position_source == "assignment_real_time_position":
            return assignment.real_time_position or cluster.center

        if self.config.start_position_source == "assignment_start_position":
            return assignment.start_position or cluster.center

        raise StrikeOrderPlannerError(
            f"Unsupported start_position_source: {self.config.start_position_source}"
        )

    def _select_dqn_action(
        self,
        observation_vector: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """使用 DQN greedy 策略选择动作。"""
        if self._dqn_agent is None:
            raise StrikeOrderPlannerError("DQN agent has not been loaded.")

        selection = self._dqn_agent.select_action(
            observation=observation_vector,
            action_mask=action_mask,
            training=False,
        )

        return int(selection.action_id)

    def _select_nearest_action(self, env: StrikeOrderEnv) -> int:
        """最近邻策略：选择距离当前位置最近的未访问目标。"""
        mask = env.get_action_mask()
        valid_action_ids = [index for index, value in enumerate(mask) if value > 0]

        if not valid_action_ids:
            raise StrikeOrderPlannerError("No valid action for nearest policy.")

        best_action_id = valid_action_ids[0]
        best_distance = float("inf")

        for action_id in valid_action_ids:
            target = env.target_cluster.targets[action_id]
            distance = env._distance(env.current_position, target.position)

            if distance < best_distance:
                best_distance = distance
                best_action_id = action_id

        return int(best_action_id)

    def _select_random_action(self, env: StrikeOrderEnv) -> int:
        """随机策略：从合法目标中随机选择一个。"""
        mask = env.get_action_mask()
        valid_action_ids = np.where(mask > 0)[0]

        if valid_action_ids.size == 0:
            raise StrikeOrderPlannerError("No valid action for random policy.")

        return int(self.rng.choice(valid_action_ids))

    def _ensure_dqn_agent(
        self,
        target_clusters: TargetClusterSet,
    ) -> None:
        """
        加载 StrikeOrder DQN 模型。

        这里用第一个 cluster 构造 sample env，得到 input_dim 和 action_dim。
        """
        if self._dqn_agent is not None:
            return

        if not target_clusters.clusters:
            raise StrikeOrderPlannerError("target_clusters must not be empty.")

        try:
            import torch

            from uav_dynamic_task_allocation.algorithms.rl.dqn.agent import DQNAgent
            from uav_dynamic_task_allocation.algorithms.rl.dqn.checkpoint import (
                DQNCheckpointManager,
            )
        except ModuleNotFoundError as exc:
            raise StrikeOrderPlannerError(
                "PyTorch is required when strike_order_planner.method is dqn/attention_dqn. "
                "Install torch, or enable allow_fallback and use nearest_neighbor/random."
            ) from exc

        sample_cluster = target_clusters.clusters[0]
        sample_env = StrikeOrderEnv(
            target_cluster=sample_cluster,
            start_position=sample_cluster.center,
            config=self.env_config,
        )
        sample_observation = sample_env.reset()

        input_dim = int(sample_observation.vector.shape[0])
        action_dim = int(sample_env.action_dim)

        network_config = self._load_dqn_network_config(
            input_dim=input_dim,
            action_dim=action_dim,
        )
        agent_config = self._load_dqn_agent_config()
        checkpoint_config = self._load_dqn_checkpoint_config()

        preferred_device = self.config.device
        device = torch.device(get_device(preferred_device))

        agent = DQNAgent(
            network_config=network_config,
            agent_config=agent_config,
            device=device,
            seed=self.config.random_seed,
        )

        checkpoint_manager = DQNCheckpointManager(
            config=checkpoint_config,
            project_root=get_project_root(),
        )

        if self.config.checkpoint_type == "best":
            load_result = checkpoint_manager.load_best(
                agent=agent,
                load_optimizer=False,
                map_location=device,
            )
        else:
            load_result = checkpoint_manager.load_latest(
                agent=agent,
                load_optimizer=False,
                map_location=device,
            )

        self.logger.info(
            "StrikeOrder DQN checkpoint loaded: "
            f"path={load_result.path}, metadata={load_result.metadata}"
        )

        self._dqn_agent = agent

    def _load_dqn_network_config(
        self,
        input_dim: int,
        action_dim: int,
    ) -> DQNNetworkConfig:
        """读取 StrikeOrder DQN 网络配置。"""
        from uav_dynamic_task_allocation.algorithms.rl.dqn.network import (
            DQNNetworkConfig,
        )

        prefix = "strike_order_dqn.network"

        hidden_dims_raw = get_config_value(
            self.full_config,
            f"{prefix}.hidden_dims",
            default=[256, 256],
        )

        network_config = DQNNetworkConfig(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dims=tuple(int(value) for value in hidden_dims_raw),
            activation=str(
                get_config_value(
                    self.full_config,
                    f"{prefix}.activation",
                    default="relu",
                )
            ),
            dropout=float(
                get_config_value(
                    self.full_config,
                    f"{prefix}.dropout",
                    default=0.0,
                )
            ),
            use_layer_norm=bool(
                get_config_value(
                    self.full_config,
                    f"{prefix}.use_layer_norm",
                    default=False,
                )
            ),
            network_type=(
                "attention_dqn"
                if self.config.method == "attention_dqn"
                else str(
                    get_config_value(
                        self.full_config,
                        f"{prefix}.network_type",
                        default="dqn",
                    )
                )
            ),
        )

        network_config.validate()
        return network_config

    def _load_dqn_agent_config(self) -> DQNAgentConfig:
        """读取 StrikeOrder DQN agent 配置。"""
        from uav_dynamic_task_allocation.algorithms.rl.dqn.agent import (
            DQNAgentConfig,
        )

        prefix = "strike_order_dqn.agent"

        agent_config = DQNAgentConfig(
            gamma=float(
                get_config_value(self.full_config, f"{prefix}.gamma", default=0.99)
            ),
            learning_rate=float(
                get_config_value(
                    self.full_config,
                    f"{prefix}.learning_rate",
                    default=5e-4,
                )
            ),
            batch_size=int(
                get_config_value(
                    self.full_config,
                    f"{prefix}.batch_size",
                    default=32,
                )
            ),
            target_update_interval=int(
                get_config_value(
                    self.full_config,
                    f"{prefix}.target_update_interval",
                    default=100,
                )
            ),
            epsilon_start=float(
                get_config_value(
                    self.full_config,
                    f"{prefix}.epsilon_start",
                    default=1.0,
                )
            ),
            epsilon_end=float(
                get_config_value(
                    self.full_config,
                    f"{prefix}.epsilon_end",
                    default=0.05,
                )
            ),
            epsilon_decay_steps=int(
                get_config_value(
                    self.full_config,
                    f"{prefix}.epsilon_decay_steps",
                    default=3000,
                )
            ),
            gradient_clip_norm=float(
                get_config_value(
                    self.full_config,
                    f"{prefix}.gradient_clip_norm",
                    default=10.0,
                )
            ),
            optimizer=str(
                get_config_value(
                    self.full_config,
                    f"{prefix}.optimizer",
                    default="adam",
                )
            ),
        )

        agent_config.validate()
        return agent_config

    def _load_dqn_checkpoint_config(self) -> DQNCheckpointConfig:
        """读取 StrikeOrder DQN checkpoint 配置。"""
        from uav_dynamic_task_allocation.algorithms.rl.dqn.checkpoint import (
            DQNCheckpointConfig,
        )

        prefix = "strike_order_dqn.checkpoint"

        checkpoint_config = DQNCheckpointConfig(
            enabled=bool(
                get_config_value(
                    self.full_config,
                    f"{prefix}.enabled",
                    default=True,
                )
            ),
            checkpoint_dir=str(
                get_config_value(
                    self.full_config,
                    f"{prefix}.checkpoint_dir",
                    default="checkpoints/strike_order_dqn",
                )
            ),
            latest_filename=str(
                get_config_value(
                    self.full_config,
                    f"{prefix}.latest_filename",
                    default="latest.pt",
                )
            ),
            best_filename=str(
                get_config_value(
                    self.full_config,
                    f"{prefix}.best_filename",
                    default="best.pt",
                )
            ),
            save_optimizer=bool(
                get_config_value(
                    self.full_config,
                    f"{prefix}.save_optimizer",
                    default=True,
                )
            ),
            save_replay_buffer=bool(
                get_config_value(
                    self.full_config,
                    f"{prefix}.save_replay_buffer",
                    default=False,
                )
            ),
        )

        checkpoint_config.validate()
        return checkpoint_config


def load_strike_order_planner_config(
    config: dict[str, Any],
) -> StrikeOrderPlannerConfig:
    """从项目总配置中读取 StrikeOrderPlannerConfig。"""
    prefix = "strike_order_planner"

    planner_config = StrikeOrderPlannerConfig(
        method=str(
            get_config_value(config, f"{prefix}.method", default="dqn")
        ),
        checkpoint_type=str(
            get_config_value(config, f"{prefix}.checkpoint_type", default="best")
        ),
        allow_fallback=bool(
            get_config_value(config, f"{prefix}.allow_fallback", default=True)
        ),
        fallback_method=str(
            get_config_value(
                config,
                f"{prefix}.fallback_method",
                default="nearest_neighbor",
            )
        ),
        start_position_source=str(
            get_config_value(
                config,
                f"{prefix}.start_position_source",
                default="cluster_center",
            )
        ),
        device=str(
            get_config_value(config, f"{prefix}.device", default="auto")
        ),
        random_seed=int(
            get_config_value(config, f"{prefix}.random_seed", default=42)
        ),
        debug_csv_path=str(
            get_config_value(
                config,
                f"{prefix}.output.debug_csv_path",
                default="outputs/intermediate/strike_order_planner.csv",
            )
        ),
    )

    planner_config.validate()
    return planner_config


def build_strike_order_planner(
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> StrikeOrderPlanner:
    """根据项目总配置构造 StrikeOrderPlanner。"""
    return StrikeOrderPlanner(
        planner_config=load_strike_order_planner_config(config),
        env_config=load_strike_order_env_config(config),
        full_config=config,
        logger=logger,
    )
