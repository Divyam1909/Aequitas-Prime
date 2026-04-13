"""
Adult Income dataset specific preprocessing.
Produces encoded feature matrix X, binary target y, and the default DatasetConfig.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from src.utils.schema import DatasetConfig

# ── Default config for Adult Income ───────────────────────────────────────────
ADULT_CONFIG = DatasetConfig(
    protected_attrs=["sex", "race"],
    target_col="income",
    privileged_values={"sex": "Male", "race": "White"},
    positive_label=1,
    dataset_name="UCI Adult Income",
)

# Categorical features to encode
CATEGORICAL_COLS = [
    "workclass", "education", "marital-status",
    "occupation", "relationship", "native-country",
]

# Continuous features — kept as-is
CONTINUOUS_COLS = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week",
]


def preprocess(
    df: pd.DataFrame,
    config: DatasetConfig = ADULT_CONFIG,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Clean and encode the Adult Income dataset.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (encoded, no target, protected attrs kept as numeric).
    y : pd.Series
        Binary target (0/1).
    df_clean : pd.DataFrame
        Full cleaned DataFrame (raw protected attr strings preserved) for
        fairness metric computation that needs original group labels.
    """
    df = df.copy()

    # ── 1. Drop rows with NaN ───────────────────────────────────────────────
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        dropped = initial_len - len(df)
        print(f"  Dropped {dropped} rows with missing values ({dropped/initial_len:.1%})")

    # ── 2. Clean income labels — strip whitespace + dots (test set artifact) ─
    df["income"] = df["income"].str.strip().str.rstrip(".")
    # Map to binary
    df["income"] = (df["income"].str.contains(">50K") | df["income"].str.contains(">50K")).astype(int)

    # ── 3. Preserve original string values for group labels ─────────────────
    df_clean = df.copy()

    # ── 4. Encode protected attributes as binary / ordinal ──────────────────
    # sex: Male=1, Female=0
    df["sex"] = (df["sex"] == "Male").astype(int)
    # race: White=1, non-White=0  (for binary DI; intersectional analysis uses df_clean)
    df["race_binary"] = (df["race"] == "White").astype(int)
    df = df.drop(columns=["race"])

    # ── 5. Ordinal-encode categoricals ──────────────────────────────────────
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[CATEGORICAL_COLS] = enc.fit_transform(df[CATEGORICAL_COLS])

    # ── 6. Build X and y ────────────────────────────────────────────────────
    feature_cols = CONTINUOUS_COLS + CATEGORICAL_COLS + ["sex", "race_binary", "education-num"]
    # deduplicate in case education-num appeared twice
    feature_cols = list(dict.fromkeys(feature_cols))
    # Ensure all requested features actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].astype(float)
    y = df[config.target_col].astype(int)

    return X, y, df_clean


def get_feature_names() -> list[str]:
    """Return the feature names produced by preprocess()."""
    cols = CONTINUOUS_COLS + CATEGORICAL_COLS + ["sex", "race_binary", "education-num"]
    return list(dict.fromkeys(cols))
