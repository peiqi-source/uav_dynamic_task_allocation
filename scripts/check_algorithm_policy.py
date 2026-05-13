"""checkalgorithm策略脚本，封装可直接运行的实验、检查或可视化流程。"""
from uav_dynamic_task_allocation.core.contracts import MissionEventType
from uav_dynamic_task_allocation.planning.algorithm_policy import (
    ScenarioMode,
    load_algorithm_policy,
)
from uav_dynamic_task_allocation.utils.config import (
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


def print_decisions(title: str, decisions, policy, logger) -> None:
    """打印策略决策。"""
    logger.info("=" * 80)
    logger.info(title)

    for item in policy.summarize_decisions(decisions):
        logger.info(item)

    logger.info(f"Config overrides: {policy.build_config_overrides(decisions)}")


def main() -> None:
    """处理main 数据相关业务逻辑。

    参数：
        无显式业务参数。

    返回：
        无返回值；通过状态变更、文件输出或日志记录体现执行结果。
    """
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    config = load_and_validate_config(config_path)
    logger = setup_logger_from_config(config)

    logger.info("Algorithm policy check started.")

    policy = load_algorithm_policy(config)

    static_initial_decisions = policy.get_initial_stage_policies(
        scenario_mode=ScenarioMode.STATIC,
    )
    print_decisions(
        title="Static initial planning decisions",
        decisions=static_initial_decisions,
        policy=policy,
        logger=logger,
    )

    dynamic_initial_decisions = policy.get_initial_stage_policies(
        scenario_mode=ScenarioMode.DYNAMIC,
    )
    print_decisions(
        title="Dynamic initial planning decisions",
        decisions=dynamic_initial_decisions,
        policy=policy,
        logger=logger,
    )

    for event_type in [
        MissionEventType.TARGET_APPEARED,
        MissionEventType.TARGET_DISAPPEARED,
        MissionEventType.ATTACK_UAV_DESTROYED,
        MissionEventType.GUIDE_UAV_DESTROYED,
        MissionEventType.COMMUNICATION_UAV_DESTROYED,
        MissionEventType.RESOURCE_SHORTAGE,
    ]:
        decisions = policy.get_replanning_decisions(
            event_type=event_type,
            scenario_mode=ScenarioMode.DYNAMIC,
        )

        print_decisions(
            title=f"Dynamic replanning decisions for event: {event_type.value}",
            decisions=decisions,
            policy=policy,
            logger=logger,
        )

    logger.info("Algorithm policy check finished successfully.")


if __name__ == "__main__":
    main()