"""check数据脚本，封装可直接运行的实验、检查或可视化流程。"""
from uav_dynamic_task_allocation.data.loaders import load_all_data
from uav_dynamic_task_allocation.utils.config import (
    get_project_root,
    load_and_validate_config,
)
from uav_dynamic_task_allocation.utils.logger import setup_logger_from_config


def print_dataframe_summary(name, df, logger) -> None:
    """
    Print basic information of a DataFrame.
    """
    logger.info(f"{name} loaded successfully.")
    logger.info(f"{name} shape: {df.shape}")
    logger.info(f"{name} columns: {list(df.columns)}")
    logger.info(f"{name} preview:")
    logger.info(f"\n{df.head().to_string(index=False)}")


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

    logger.info("Data check started.")

    data = load_all_data(config)

    print_dataframe_summary("UAV data", data["uav"], logger)
    print_dataframe_summary("Target data", data["target"], logger)
    print_dataframe_summary(
        "Battlefield target data",
        data["battlefield_target"],
        logger,
    )

    logger.info("Data validation finished successfully.")


if __name__ == "__main__":
    main()