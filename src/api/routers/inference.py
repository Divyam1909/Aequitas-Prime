"""
/predict route — live prediction with Counterfactual Ghost firewall + SHAP.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/predict", tags=["inference"])

# Module-level model cache — loaded once at startup via lifespan in main.py
_model_cache: dict[str, Any] = {}


def set_model(model, feature_names: list[str], config):
    _model_cache["model"]         = model
    _model_cache["feature_names"] = feature_names
    _model_cache["config"]        = config


class PredictRequest(BaseModel):
    features: dict[str, Any]
    run_counterfactual: bool = True
    run_shap: bool = True


class PredictResponse(BaseModel):
    decision: str
    confidence: float
    risk_level: str
    counterfactual: dict | None = None
    shap_top_features: list[dict] | None = None


@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest):
    if "model" not in _model_cache:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the pipeline first.")

    model         = _model_cache["model"]
    feature_names = _model_cache["feature_names"]
    config        = _model_cache["config"]

    # Build input row
    row_dict = {f: req.features.get(f, 0) for f in feature_names}
    X_row = pd.DataFrame([row_dict])

    # Prediction
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_row)[0]
        pred  = int(np.argmax(proba))
        conf  = float(proba[pred])
    else:
        pred = int(model.predict(X_row)[0])
        conf = 1.0

    decision = "Approved" if pred == config.positive_label else "Denied"

    # Counterfactual Ghost
    cf_out = None
    if req.run_counterfactual:
        from src.bias_engine.counterfactual import run_counterfactual_check
        cf = run_counterfactual_check(model, row_dict, config, feature_names)
        cf_out = {
            "flipped_attr":              cf.protected_attr,
            "original_value":            cf.original_value,
            "flipped_value":             cf.flipped_value,
            "counterfactual_decision":   cf.counterfactual_decision,
            "confidence_delta":          round(cf.confidence_delta, 3),
            "risk_level":                cf.risk_level,
            "message":                   cf.message,
        }

    # SHAP top features
    shap_out = None
    if req.run_shap:
        try:
            from src.explainability.shap_explainer import build_explainer, explain_single
            if "explainer" not in _model_cache:
                _model_cache["explainer"] = build_explainer(model, X_row)
            exp  = _model_cache["explainer"]
            local = explain_single(exp, X_row, feature_names, pred)
            shap_out = [
                {"feature": f, "value": v, "shap": round(s, 4)}
                for f, v, s in local.top_drivers(5)
            ]
        except Exception as e:
            shap_out = [{"error": str(e)}]

    risk = cf_out["risk_level"] if cf_out else "CLEAR"

    return PredictResponse(
        decision=decision,
        confidence=round(conf, 3),
        risk_level=risk,
        counterfactual=cf_out,
        shap_top_features=shap_out,
    )
