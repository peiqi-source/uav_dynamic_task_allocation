# 为了让类型标注更灵活
from __future__ import annotations
from pathlib import Path
# 例如：dict[str, Any] 这是一个字典，key是字符串，value可以是任何类型
from typing import Any
import yaml

class ConfigError(Exception):
    """
    配置文件相关的自定义错误。
    如：raise ConfigError("Config file not found")
    """


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    读取 YAML 配置文件。

    Args:
        config_path: 配置文件路径，例如 configs/default.yaml

    Returns:
        读取后的配置字典。

    Raises:
        ConfigError: 当配置文件不存在、格式错误或内容为空时抛出。
    """
    # 把传进来的路径统一转成 Path 对象，后面统一用 Path 处理
    path = Path(config_path)

    # 检查配置文件是否存在
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    # 只允许读取 .yaml 或 .yml 文件。
    if path.suffix not in {".yaml", ".yml"}:
        raise ConfigError(f"Config file must be a YAML file: {path}")
    # 读取 YAML 文件，它会把 YAML 文件变成 Python 字典
    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config: {path}") from exc
    # 如果配置文件是空的，yaml.safe_load 会返回 None
    if config is None:
        raise ConfigError(f"Config file is empty: {path}")
    # 检查读出来的配置是不是字典
    if not isinstance(config, dict):
        raise ConfigError(f"Config file must contain a YAML dictionary: {path}")

    return config



def validate_config(config: dict[str, Any]) -> None:
    """
    检查配置文件里是否包含必要的大模块。
    """
    required_sections = [
        "project",
        "experiment",
        "paths",
        "data",
        "env",
        "algorithm",
        "training",
        "logging",
        "output",
    ]

    missing_sections = [
        section for section in required_sections if section not in config
    ]

    if missing_sections:
        raise ConfigError(
            "Missing required config sections: "
            + ", ".join(missing_sections)
        )


def load_and_validate_config(config_path: str | Path) -> dict[str, Any]:
    """
    读取配置文件，并检查必要字段是否存在。
    """
    config = load_config(config_path)
    validate_config(config)
    return config


def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件位置是：
    src/uav_dynamic_task_allocation/utils/config.py

    所以：
    parents[0] = utils
    parents[1] = uav_dynamic_task_allocation
    parents[2] = src
    parents[3] = 项目根目录
    """
    return Path(__file__).resolve().parents[3]


def resolve_path(path_value: str | Path, project_root: str | Path | None = None) -> Path:
    """
    把配置文件里的相对路径转换成绝对路径。

    例如配置文件里写：
        data/raw/UAV.csv

    转换后变成：
        F:/项目/专利/uav_dynamic_task_allocation/data/raw/UAV.csv
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    root = Path(project_root) if project_root is not None else get_project_root()
    return root / path



def get_config_value(
    config: dict[str, Any],
    key_path: str,
    default: Any | None = None,
) -> Any:
    """
    用点号路径读取嵌套配置。
    如果配置找不到，也不会直接报错，而是返回默认值。

    例如：
        get_config_value(config, "training.episodes")

    等价于：
        config["training"]["episodes"]
    """
    current: Any = config

    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current