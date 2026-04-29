from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from uav_dynamic_task_allocation.utils.config import (
    get_config_value,
    get_project_root,
    resolve_path,
)

# 把字符串转换成 logging 级别
def get_log_level(level: str | int) -> int:
    """
    Convert a logging level from string or integer to logging module level.

    Examples:
        "INFO" -> logging.INFO
        "DEBUG" -> logging.DEBUG
    """
    if isinstance(level, int):
        return level

    level = level.upper()

    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if level not in level_mapping:
        raise ValueError(
            f"Invalid log level: {level}. "
            f"Available levels: {list(level_mapping.keys())}"
        )

    return level_mapping[level]

# 自动创建日志文件路径
def create_log_file_path(
    log_dir: str | Path,
    experiment_name: str,
    project_root: str | Path | None = None,
) -> Path:
    """
    Create a timestamped log file path.

    Example:
        logs/debug_experiment_20260429_153000.log
    """
    root = Path(project_root) if project_root is not None else get_project_root()
    log_dir_path = resolve_path(log_dir, root)

    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{experiment_name}_{timestamp}.log"

    return log_dir_path / log_file_name

# 真正创建 logger
def setup_logger(
    name: str = "uav_dynamic_task_allocation",
    log_file: str | Path | None = None,
    level: str | int = "INFO",
    use_console: bool = True,
) -> logging.Logger:
    """
    Set up a project logger.

    Args:
        name: Logger name.
        log_file: Optional log file path.
        level: Logging level, such as INFO, DEBUG, WARNING.
        use_console: Whether to print logs to terminal.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(get_log_level(level))

    # Avoid duplicate logs when setup_logger is called multiple times.
    logger.handlers.clear()
    logger.propagate = False

    # formatter 负责控制格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console handler 负责输出到终端
    if use_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(get_log_level(level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # file handler 负责写入日志文件
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(get_log_level(level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# 从配置文件创建 logger
def setup_logger_from_config(
    config: dict[str, Any],
    name: str = "uav_dynamic_task_allocation",
) -> logging.Logger:
    """
    Set up logger using project configuration.

    This function reads:
        logging.level
        logging.save_log_file
        paths.log_dir
        experiment.name

    from the config dictionary.
    """
    project_root = get_project_root()

    level = get_config_value(config, "logging.level", default="INFO")
    save_log_file = get_config_value(config, "logging.save_log_file", default=True)
    log_dir = get_config_value(config, "paths.log_dir", default="logs")
    experiment_name = get_config_value(
        config,
        "experiment.name",
        default="default_experiment",
    )

    log_file = None
    if save_log_file:
        log_file = create_log_file_path(
            log_dir=log_dir,
            experiment_name=experiment_name,
            project_root=project_root,
        )

    logger = setup_logger(
        name=name,
        log_file=log_file,
        level=level,
        use_console=True,
    )

    logger.info("Logger initialized.")
    if log_file is not None:
        logger.info(f"Log file: {log_file}")

    return logger