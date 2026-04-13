"""
Model persistence — save/load trained models with metadata.
Uses joblib (handles sklearn + Fairlearn ensembles).
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Any
import joblib
import numpy as np

from src.utils.schema import DatasetConfig


MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def _json_safe(obj: Any) -> Any:
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def save_artifact(
    model: Any,
    model_type: str,
    config: DatasetConfig,
    metrics: dict,
    feature_names: list[str],
    name: str | None = None,
) -> Path:
    """
    Save a model + metadata to models/.
    Returns the path to the saved .joblib file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    slug = name or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = MODELS_DIR / f"{slug}.joblib"
    meta_path  = MODELS_DIR / f"{slug}_meta.json"

    joblib.dump(model, model_path, protocol=4)

    metadata = {
        "model_type":    model_type,
        "dataset_name":  config.dataset_name,
        "protected_attrs": config.protected_attrs,
        "target_col":    config.target_col,
        "feature_names": feature_names,
        "train_date":    datetime.now().isoformat(),
        "metrics":       _json_safe(metrics),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"  Saved model: {model_path.name}")
    return model_path


def load_artifact(name: str) -> tuple[Any, dict]:
    """
    Load a model and its metadata by name (without extension).
    Returns (model, metadata_dict).
    """
    model_path = MODELS_DIR / f"{name}.joblib"
    meta_path  = MODELS_DIR / f"{name}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model    = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, metadata


def list_artifacts() -> list[str]:
    """Return names of all saved model artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return [p.stem for p in MODELS_DIR.glob("*.joblib")]
