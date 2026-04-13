"""
Central schema definitions used by every module in the pipeline.
DatasetConfig drives all fairness logic — it describes how to interpret a dataset.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    """
    Describes a binary-classification dataset for fairness analysis.

    Attributes
    ----------
    protected_attrs : list[str]
        Column names of protected attributes, e.g. ["sex", "race"].
    target_col : str
        Binary target column name, e.g. "income".
    privileged_values : dict[str, Any]
        The value in each protected column that belongs to the privileged group,
        e.g. {"sex": "Male", "race": "White"}.
    positive_label : Any
        The target value that represents a positive outcome, e.g. ">50K" or 1.
    dataset_name : str
        Human-readable name for reports.
    """
    protected_attrs: list[str]
    target_col: str
    privileged_values: dict[str, Any]
    positive_label: Any
    dataset_name: str = "Dataset"

    def primary_protected_attr(self) -> str:
        """Returns the first protected attribute (used where only one is needed)."""
        return self.protected_attrs[0]

    def is_privileged(self, row: dict[str, Any], attr: str) -> bool:
        return row.get(attr) == self.privileged_values.get(attr)
