"""
Generic CSV loader with validation.
Works for any dataset — not just Adult Income.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.utils.schema import DatasetConfig


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file, strip string whitespace, and return a clean DataFrame.
    Handles both comma and semicolon delimiters.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, sep=None, engine="python", na_values=["?", " ?", "NA", ""])
    # Strip leading/trailing whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    return df


def validate_columns(df: pd.DataFrame, config: DatasetConfig) -> None:
    """
    Raise ValueError if required columns are missing or misconfigured.
    """
    missing = []
    for col in config.protected_attrs + [config.target_col]:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(
            f"Required columns missing from dataset: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    for attr, priv_val in config.privileged_values.items():
        unique_vals = df[attr].dropna().unique().tolist()
        if priv_val not in unique_vals:
            raise ValueError(
                f"Privileged value '{priv_val}' not found in column '{attr}'.\n"
                f"Unique values: {unique_vals}"
            )

    # Note: positive_label may be the post-encoding value (e.g. 1) while raw data
    # still holds string labels. Validation of label mapping is handled in preprocessor.
