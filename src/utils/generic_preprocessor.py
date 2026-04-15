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
            # Skip continuous numeric columns — e.g. an "age" column with 60+
            # distinct integer values cannot be meaningfully split into a
            # privileged/unprivileged group by a single value.
            n_unique = df[col].nunique()
            if df[col].dtype != object and n_unique > 20:
                continue
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
    # Sync binarized target into df_clean so the fairness engine compares numeric
    # 0/1 values against config.positive_label, which we also update to 1.
    df_clean[config.target_col] = df[config.target_col]
    # CRITICAL: update positive_label to the numeric value (1) so the detector's
    # comparisons (priv_y == config.positive_label) work on the binarized column.
    config.positive_label = 1

    # ── 5. Encode protected attrs as binary 0/1 (keep original column names) ──
    # df_clean retains original string values for group-split logic in detector.py
    for attr in config.protected_attrs:
        if attr not in df.columns:
            continue
        priv_val = config.privileged_values[attr]
        df[attr] = (df[attr].astype(str) == str(priv_val)).astype(int)
        # df_clean[attr] is untouched (still has "Male"/"Female" etc.)

    # ── 6. Ordinal-encode remaining categorical / non-numeric columns ─────────
    # Use is_numeric_dtype instead of dtype == object so that Arrow-backed
    # StringDtype columns (e.g. time strings like "17:02:00") are also caught.
    cat_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and c != config.target_col
    ]
    if cat_cols:
        # Cast to plain object dtype (not Arrow StringDtype) so OrdinalEncoder's
        # float output can be assigned back without a string-cast error.
        cat_df = df[cat_cols].astype(str).astype(object)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded = enc.fit_transform(cat_df)
        df[cat_cols] = pd.DataFrame(encoded, index=df.index, columns=cat_cols)

    # ── 7. Build feature matrix X and target y ────────────────────────────────
    feature_cols = [c for c in df.columns if c != config.target_col]

    # Catch-all: coerce any column that still can't cast to float (e.g. mixed
    # types, object columns that slipped through) rather than crashing.
    for col in feature_cols:
        try:
            df[col].astype(float)
        except (ValueError, TypeError):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

    X = df[feature_cols].astype(float)
    y = df[config.target_col].astype(int)

    return X, y, df_clean
