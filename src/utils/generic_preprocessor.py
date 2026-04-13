"""
Generic dataset preprocessor — works with any binary classification CSV.
Companion to adult_preprocessor.py (which is Adult-Income specific).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from src.utils.schema import DatasetConfig

# Keywords used to auto-detect protected attributes by column name
PROTECTED_KEYWORDS = [
    "sex", "gender", "race", "ethnic", "religion",
    "nation", "disab", "marital", "color", "pregnan",
    "caste", "orient", "citizen", "age",
]

# Keywords used to auto-detect the target/label column
TARGET_KEYWORDS = [
    "label", "target", "income", "outcome", "class",
    "predict", "result", "approved", "hired", "decision",
    "default", "fraud", "churn", "y",
]


def auto_detect_protected_attrs(
    df: pd.DataFrame, target_col: str | None = None
) -> list[str]:
    """
    Heuristically detect candidate protected attribute columns.

    Strategy:
    1. Column name matches a known protected-attribute keyword.
    2. Low-cardinality categorical column (2–20 unique values).
    """
    candidates: list[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower().replace("-", "_").replace(" ", "_")

        # Keyword match (highest confidence)
        if any(kw in col_lower for kw in PROTECTED_KEYWORDS):
            candidates.append(col)
            continue

        # Cardinality heuristic: small number of distinct values → likely categorical/demographic
        n_unique = df[col].nunique()
        if df[col].dtype == object and 2 <= n_unique <= 20:
            candidates.append(col)

    return list(dict.fromkeys(candidates))  # preserve order, deduplicate


def auto_detect_target(df: pd.DataFrame) -> str | None:
    """
    Heuristically guess the target column.
    Falls back to the last column if no keyword match found.
    """
    for col in reversed(df.columns.tolist()):
        col_lower = col.lower().replace("-", "_").replace(" ", "_")
        if any(kw in col_lower for kw in TARGET_KEYWORDS):
            return col
    return df.columns[-1]


def preprocess_generic(
    df: pd.DataFrame,
    config: DatasetConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Generic preprocessor for any binary classification dataset.

    Parameters
    ----------
    df : raw DataFrame (any CSV)
    config : DatasetConfig with protected_attrs, target_col, privileged_values,
             positive_label already populated by the user.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all categoricals ordinal-encoded, protected attrs kept as
        binary 0/1 under their original column names, no target column).
    y : pd.Series
        Binary target (1 = positive_label, 0 = otherwise).
    df_clean : pd.DataFrame
        Full cleaned DataFrame with ORIGINAL string labels preserved — needed by
        the fairness engine which splits groups by string value.
    """
    df = df.copy()

    # ── 1. Drop rows with missing values in key columns ───────────────────────
    key_cols = [config.target_col] + [
        a for a in config.protected_attrs if a in df.columns
    ]
    df = df.dropna(subset=key_cols)

    # ── 2. Strip leading/trailing whitespace from string columns ──────────────
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # ── 3. Preserve original string values for fairness engine ────────────────
    df_clean = df.copy()

    # ── 4. Binarize target: positive_label → 1, anything else → 0 ────────────
    pos = config.positive_label
    df[config.target_col] = (df[config.target_col].astype(str) == str(pos)).astype(int)
    # Sync binarized target into df_clean (fairness engine uses numeric target)
    df_clean[config.target_col] = df[config.target_col]

    # ── 5. Encode protected attrs as binary 0/1 (keep original column names) ──
    # df_clean retains original string values for group-split logic in detector.py
    for attr in config.protected_attrs:
        if attr not in df.columns:
            continue
        priv_val = config.privileged_values[attr]
        df[attr] = (df[attr].astype(str) == str(priv_val)).astype(int)
        # df_clean[attr] is untouched (still has "Male"/"Female" etc.)

    # ── 6. Ordinal-encode remaining categorical columns ───────────────────────
    cat_cols = [
        c for c in df.columns
        if df[c].dtype == object and c != config.target_col
    ]
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols])

    # ── 7. Build feature matrix X and target y ────────────────────────────────
    feature_cols = [c for c in df.columns if c != config.target_col]
    X = df[feature_cols].astype(float)
    y = df[config.target_col].astype(int)

    return X, y, df_clean
