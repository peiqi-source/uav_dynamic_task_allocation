"""数据模块中的loaders 数据实现。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from uav_dynamic_task_allocation.data.schemas import (
    TARGET_SCHEMA,
    UAV_SCHEMA,
    DataSchema,
)
from uav_dynamic_task_allocation.data.validators import (
    validate_target_data,
    validate_uav_data,
)
from uav_dynamic_task_allocation.utils.config import resolve_path


class DataLoadError(Exception):
    """Custom exception for data loading errors."""


def read_csv_with_auto_encoding(
    file_path: str | Path,
    encodings: Iterable[str] = ("utf-8-sig", "utf-8", "gbk", "gb18030"),
) -> pd.DataFrame:
    """
    Read a CSV file with multiple possible encodings.

    This is useful because some project files use Chinese GBK encoding,
    while others use UTF-8 with BOM.

    Args:
        file_path: Path to CSV file.
        encodings: Candidate encodings.

    Returns:
        Loaded DataFrame.

    Raises:
        DataLoadError: If the file does not exist or cannot be decoded.
    """
    path = Path(file_path)

    if not path.exists():
        raise DataLoadError(f"CSV file not found: {path}")

    errors: list[str] = []

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    raise DataLoadError(
        f"Failed to read CSV file with candidate encodings: {path}\n"
        + "\n".join(errors)
    )


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing spaces and BOM characters.
    """
    df = df.copy()
    df.columns = [
        str(column).strip().replace("\ufeff", "")
        for column in df.columns
    ]
    return df


def standardize_columns(
    df: pd.DataFrame,
    schema: DataSchema,
) -> pd.DataFrame:
    """
    Rename raw columns to standard column names according to schema.
    """
    df = clean_column_names(df)
    df = df.rename(columns=schema.column_aliases)
    return df


def convert_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """
    Convert selected columns to numeric values.
    """
    df = df.copy()

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def load_uav_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load, standardize, and validate UAV data.

    Returns:
        Standardized UAV DataFrame with columns:
        uav_id, x, y, uav_type, work_range, attack_power
    """
    df = read_csv_with_auto_encoding(file_path)
    df = standardize_columns(df, UAV_SCHEMA)
    df = convert_numeric_columns(df, UAV_SCHEMA.numeric_columns)

    validate_uav_data(df)

    return df


def load_target_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load, standardize, and validate target data.

    Returns:
        Standardized target DataFrame with columns:
        target_id, x, y, target_type, defense, significance
    """
    df = read_csv_with_auto_encoding(file_path)
    df = standardize_columns(df, TARGET_SCHEMA)
    df = convert_numeric_columns(df, TARGET_SCHEMA.numeric_columns)

    validate_target_data(df, data_name="Target data")

    return df


def load_all_data(config: dict) -> dict[str, pd.DataFrame]:
    """
    Load all core data files based on project config.

    Args:
        config: Project configuration dictionary.

    Returns:
        Dictionary containing UAV, target, and battlefield target data.
    """
    from uav_dynamic_task_allocation.utils.config import get_config_value

    uav_file = resolve_path(get_config_value(config, "data.uav_file"))
    target_file = resolve_path(get_config_value(config, "data.target_file"))
    battlefield_file = resolve_path(
        get_config_value(config, "data.battlefield_file")
    )

    uav_data = load_uav_data(uav_file)
    target_data = load_target_data(target_file)


    return {
        "uav": uav_data,
        "target": target_data,
    }
