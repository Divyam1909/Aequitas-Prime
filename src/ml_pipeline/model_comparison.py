"""
Multi-model comparison — trains RF, LR, and XGB baselines on the same split
and returns a tidy DataFrame of performance + fairness metrics side-by-side.

Used by the Surgeon tab's "Compare All Base Models" panel.
"""
from __future__ import annotations
import pandas as pd

from src.utils.schema import DatasetConfig
from src.ml_pipeline.trainer import train_baseline, MODEL_DISPLAY
from src.ml_pipeline.evaluator import evaluate


def compare_base_models(
    X: pd.DataFrame,
    y: pd.Series,
    df_clean: pd.DataFrame,
    config: DatasetConfig,
    models: tuple[str, ...] = ("rf", "lr", "xgb"),
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train each model in `models` and return a comparison DataFrame.

    Columns: Model | Accuracy | F1 | ROC-AUC | DI | SPD | EOD
    One row per model.
    """
    primary = config.primary_protected_attr()
    rows = []

    for key in models:
        tr = train_baseline(X, y, config,
                            base_model=key,
                            test_size=test_size,
                            random_state=random_state)
        ev = evaluate(tr, df_clean, config)
        perf = ev.performance
        fair = ev.fairness.get(primary)

        row: dict = {
            "Model":    MODEL_DISPLAY.get(key, key),
            "Accuracy": round(perf.accuracy, 4),
            "F1":       round(perf.f1, 4),
            "ROC-AUC":  round(perf.roc_auc, 4) if perf.roc_auc is not None else None,
        }
        if fair is not None:
            row["DI"]  = round(fair.disparate_impact, 4)
            row["SPD"] = round(fair.statistical_parity_difference, 4)
            row["EOD"] = round(fair.equalized_odds_difference, 4)
        else:
            row["DI"] = row["SPD"] = row["EOD"] = None

        rows.append(row)

    return pd.DataFrame(rows)
