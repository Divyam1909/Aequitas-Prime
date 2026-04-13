"""
Shared pytest fixtures for Aequitas Prime test suite.

Two synthetic datasets:
  biased_df  — strong demographic bias (DI ≈ 0.45, unprivileged group favoured far less)
  fair_df    — balanced outcomes across groups (DI ≈ 1.0)

Both are pure-Python / numpy — no file I/O required.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.utils.schema import DatasetConfig


RNG = np.random.default_rng(42)
N   = 600


def _make_df(priv_approval: float, unpriv_approval: float, n: int = N) -> pd.DataFrame:
    """
    Build a synthetic binary-classification DataFrame with one protected attribute.

    Columns: age, education_num, hours_per_week, sex (Male/Female), income (0/1)
    'Male' is privileged; approval rate differs between groups as specified.
    """
    half = n // 2

    # Privileged group (Male)
    male_income = RNG.binomial(1, priv_approval, half)
    male_df = pd.DataFrame({
        "age":            RNG.integers(25, 65, half),
        "education_num":  RNG.integers(8, 16, half),
        "hours_per_week": RNG.integers(35, 55, half),
        "sex":            "Male",
        "income":         male_income,
    })

    # Unprivileged group (Female)
    female_income = RNG.binomial(1, unpriv_approval, n - half)
    female_df = pd.DataFrame({
        "age":            RNG.integers(25, 65, n - half),
        "education_num":  RNG.integers(8, 16, n - half),
        "hours_per_week": RNG.integers(35, 55, n - half),
        "sex":            "Female",
        "income":         female_income,
    })

    df = pd.concat([male_df, female_df], ignore_index=True)
    df["income"] = df["income"].astype(int)
    return df


@pytest.fixture(scope="session")
def biased_df() -> pd.DataFrame:
    """Strongly biased dataset: males 70% approval, females 30% approval → DI ≈ 0.43."""
    return _make_df(priv_approval=0.70, unpriv_approval=0.30)


@pytest.fixture(scope="session")
def fair_df() -> pd.DataFrame:
    """Fair dataset: both groups ~50% approval → DI ≈ 1.0."""
    return _make_df(priv_approval=0.50, unpriv_approval=0.50)


@pytest.fixture(scope="session")
def base_config() -> DatasetConfig:
    return DatasetConfig(
        protected_attrs=["sex"],
        target_col="income",
        privileged_values={"sex": "Male"},
        positive_label=1,
        dataset_name="Synthetic",
    )


@pytest.fixture(scope="session")
def biased_clean(biased_df: pd.DataFrame) -> pd.DataFrame:
    """df_clean version of biased_df (string labels preserved, target numeric)."""
    return biased_df.copy()


@pytest.fixture(scope="session")
def preprocessed_biased(biased_df: pd.DataFrame, base_config: DatasetConfig):
    """Returns (X, y, df_clean) after running the generic preprocessor."""
    from src.utils.generic_preprocessor import preprocess_generic
    return preprocess_generic(biased_df, base_config)
