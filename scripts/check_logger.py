"""check日志器脚本，封装可直接运行的实验、检查或可视化流程。"""
from uav_dynamic_task_allocation.utils.config import (
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


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

    logger.debug("This is a DEBUG message.")
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")

    logger.info("Logger check finished successfully.")


if __name__ == "__main__":
    main()