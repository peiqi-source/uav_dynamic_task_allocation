from __future__ import annotations

import pandas as pd

from uav_dynamic_task_allocation.data.schemas import DataSchema


class DataValidationError(Exception):
    """Custom exception for data validation errors."""


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    data_name: str,
) -> None:
    """
    Check whether required columns exist.

    Args:
        df: Input DataFrame.
        required_columns: Columns that must exist.
        data_name: Name of the data table, used in error messages.
    """
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise DataValidationError(
            f"{data_name} is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
    data_name: str,
) -> None:
    """
    Check whether numeric columns can be converted to numbers.
    """
    for column in numeric_columns:
        if column not in df.columns:
            continue

        converted = pd.to_numeric(df[column], errors="coerce")

        if converted.isna().any():
            invalid_rows = df[converted.isna()].index.tolist()
            raise DataValidationError(
                f"{data_name}.{column} contains non-numeric values "
                f"at rows: {invalid_rows[:10]}"
            )


def validate_no_missing_values(
    df: pd.DataFrame,
    columns: list[str],
    data_name: str,
) -> None:
    """
    Check missing values in important columns.
    """
    for column in columns:
        if column not in df.columns:
            continue

        if df[column].isna().any():
            missing_rows = df[df[column].isna()].index.tolist()
            raise DataValidationError(
                f"{data_name}.{column} contains missing values "
                f"at rows: {missing_rows[:10]}"
            )


def validate_non_negative_columns(
    df: pd.DataFrame,
    columns: list[str],
    data_name: str,
) -> None:
    """
    Check whether selected numeric columns are non-negative.
    """
    for column in columns:
        if column not in df.columns:
            continue

        if (df[column] < 0).any():
            invalid_rows = df[df[column] < 0].index.tolist()
            raise DataValidationError(
                f"{data_name}.{column} contains negative values "
                f"at rows: {invalid_rows[:10]}"
            )


def validate_uav_data(df: pd.DataFrame) -> None:
    """
    Validate standardized UAV data.
    """
    validate_required_columns(
        df=df,
        required_columns=[
            "uav_id",
            "x",
            "y",
            "uav_type",
            "work_range",
            "attack_power",
        ],
        data_name="UAV data",
    )

    validate_no_missing_values(
        df=df,
        columns=[
            "uav_id",
            "x",
            "y",
            "uav_type",
            "work_range",
            "attack_power",
        ],
        data_name="UAV data",
    )

    validate_numeric_columns(
        df=df,
        numeric_columns=[
            "uav_id",
            "x",
            "y",
            "uav_type",
            "work_range",
            "attack_power",
        ],
        data_name="UAV data",
    )

    validate_non_negative_columns(
        df=df,
        columns=[
            "uav_id",
            "uav_type",
            "work_range",
            "attack_power",
        ],
        data_name="UAV data",
    )

    valid_uav_types = {1, 2, 3}
    invalid_types = set(df["uav_type"].unique()) - valid_uav_types

    if invalid_types:
        raise DataValidationError(
            f"UAV data contains invalid uav_type values: {invalid_types}. "
            "Valid values are: 1=guide, 2=communication, 3=attack."
        )


def validate_target_data(df: pd.DataFrame, data_name: str = "Target data") -> None:
    """
    Validate standardized target data.
    """
    validate_required_columns(
        df=df,
        required_columns=[
            "target_id",
            "x",
            "y",
            "target_type",
            "defense",
            "significance",
        ],
        data_name=data_name,
    )

    validate_no_missing_values(
        df=df,
        columns=[
            "target_id",
            "x",
            "y",
            "target_type",
            "defense",
            "significance",
        ],
        data_name=data_name,
    )

    validate_numeric_columns(
        df=df,
        numeric_columns=[
            "target_id",
            "x",
            "y",
            "target_type",
            "defense",
            "significance",
        ],
        data_name=data_name,
    )

    validate_non_negative_columns(
        df=df,
        columns=[
            "target_id",
            "target_type",
            "defense",
            "significance",
        ],
        data_name=data_name,
    )


def validate_dataframe_by_schema(
    df: pd.DataFrame,
    schema: DataSchema,
    data_name: str,
) -> None:
    """
    Generic validation based on a DataSchema.
    """
    validate_required_columns(df, schema.required_columns, data_name)
    validate_no_missing_values(df, schema.required_columns, data_name)
    validate_numeric_columns(df, schema.numeric_columns, data_name)