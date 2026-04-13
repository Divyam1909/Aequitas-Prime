"""
/audit routes — CSV upload → full pipeline → AuditResult.
"""

from __future__ import annotations
import io
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from src.utils.data_loader import load_csv, validate_columns
from src.utils.adult_preprocessor import ADULT_CONFIG
from src.ml_pipeline.pipeline import run_full_pipeline

router = APIRouter(prefix="/audit", tags=["audit"])

# In-memory job store (replace with Redis for production)
_jobs: dict[str, dict[str, Any]] = {}


def _serialize_result(result) -> dict:
    """Convert AuditResult to a JSON-safe dict for the API response."""
    primary = result.config.primary_protected_attr()

    def fmt_metrics(eval_result):
        if not eval_result:
            return None
        m = eval_result.fairness.get(primary)
        return {
            "accuracy":          round(eval_result.performance.accuracy, 4),
            "f1":                round(eval_result.performance.f1, 4),
            "disparate_impact":  round(m.disparate_impact, 3) if m else None,
            "spd":               round(m.statistical_parity_diff, 3) if m else None,
            "eod":               round(m.equalized_odds_diff, 3) if m else None,
            "eopd":              round(m.equal_opportunity_diff, 3) if m else None,
            "severity":          m.severity if m else "UNKNOWN",
        }

    proxies = []
    for r_list in result.proxy_scan.values():
        for r in r_list:
            if r.risk_level in ("HIGH", "MEDIUM"):
                proxies.append({
                    "feature":      r.feature,
                    "protected_attr": r.protected_attr,
                    "mutual_info":  round(r.mutual_info, 4),
                    "risk_level":   r.risk_level,
                    "action":       r.action,
                })

    return {
        "dataset_name":       result.dataset_name,
        "n_rows":             result.n_rows,
        "n_features":         result.n_features,
        "baseline":           fmt_metrics(result.baseline_eval),
        "preprocessed":       fmt_metrics(result.preprocessed_eval),
        "inprocessed":        fmt_metrics(result.inprocessed_eval),
        "proxy_flags":        proxies[:10],
        "comparison_table":   result.comparison_df.to_dict(orient="records") if result.comparison_df is not None else [],
    }


@router.post("/upload")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    run_inprocessing: bool = False,
):
    """
    Upload a CSV and kick off the full bias audit pipeline.
    Returns a job_id immediately; poll /audit/status/{job_id} for completion.
    """
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "result": None, "error": None}

    contents = await file.read()

    def run_job():
        try:
            df = pd.read_csv(io.BytesIO(contents), sep=None, engine="python",
                             na_values=["?", " ?", "NA", ""])
            str_cols = df.select_dtypes(include="object").columns
            df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

            result = run_full_pipeline(df, ADULT_CONFIG, run_inprocessing=run_inprocessing)
            _jobs[job_id]["result"] = _serialize_result(result)
            _jobs[job_id]["status"] = "done"
        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(e)

    background_tasks.add_task(run_job)
    return {"job_id": job_id, "status": "running"}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job["status"], "error": job.get("error")}


@router.get("/result/{job_id}")
def get_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail=f"Job status: {job['status']}")
    return job["result"]
