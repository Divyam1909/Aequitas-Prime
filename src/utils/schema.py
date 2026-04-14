"""
Central schema definitions used by every module in the pipeline.
DatasetConfig drives all fairness logic — it describes how to interpret a dataset.

V2 additions:
  - task_type: "binary" | "multiclass" | "regression"
  - Helper predicates: is_binary(), is_multiclass(), is_regression()
  - label_encodings: optional mapping of original string labels to encoded ints
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    """
    Describes a dataset for fairness analysis.

    Attributes
    ----------
    protected_attrs : list[str]
        Column names of protected attributes, e.g. ["sex", "race"].
    target_col : str
        Target column name, e.g. "income".
    privileged_values : dict[str, Any]
        The value in each protected column that belongs to the privileged group,
        e.g. {"sex": "Male", "race": "White"}.
    positive_label : Any
        The target value representing a positive outcome, e.g. ">50K" or 1.
        For multiclass / regression this may be None.
    dataset_name : str
        Human-readable name for reports.
    task_type : str
        "binary"      — standard binary classification (default, V1-compatible)
        "multiclass"  — multi-class classification
        "regression"  — continuous regression target
    label_encodings : dict[str, dict[str, int]]
        Optional reverse-lookup: original string labels → encoded integers per
        protected attribute.  Populated by the preprocessor so the counterfactual
        engine can show "Male → Female" instead of "1 → 0".
        Format: {"sex": {"Male": 1, "Female": 0}, "race": {"White": 1, ...}}
    """
    protected_attrs: list[str]
    target_col: str
    privileged_values: dict[str, Any]
    positive_label: Any
    dataset_name: str = "Dataset"
    task_type: str = "binary"
    # str_label → encoded_int, populated during preprocessing
    label_encodings: dict[str, dict[str, int]] = field(default_factory=dict)

    # ── Convenience predicates ─────────────────────────────────────────────────
    def is_binary(self) -> bool:
        return self.task_type == "binary"

    def is_multiclass(self) -> bool:
        return self.task_type == "multiclass"

    def is_regression(self) -> bool:
        return self.task_type == "regression"

    def primary_protected_attr(self) -> str:
        """Returns the first protected attribute (used where only one is needed)."""
        return self.protected_attrs[0]

    def is_privileged(self, row: dict[str, Any], attr: str) -> bool:
        return row.get(attr) == self.privileged_values.get(attr)

    def decode_label(self, attr: str, encoded_val: int | float) -> str:
        """
        Convert an encoded integer back to its original string label.
        Falls back to str(encoded_val) if no encoding map is available.
        """
        enc = self.label_encodings.get(attr, {})
        # Build reverse map: encoded_int → str_label
        rev = {v: k for k, v in enc.items()}
        return rev.get(int(encoded_val), str(encoded_val))

    def all_encoded_values(self, attr: str) -> list[int]:
        """
        Return all valid encoded integer values for a protected attribute.
        Returns [0, 1] if no encoding map is registered.
        """
        enc = self.label_encodings.get(attr, {})
        if enc:
            return sorted(set(enc.values()))
        return [0, 1]
