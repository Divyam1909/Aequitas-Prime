"""
/report route — generate and return the PDF compliance report.
"""

from __future__ import annotations
import io
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/report", tags=["report"])

_audit_cache: dict[str, Any] = {}


def set_audit_result(result):
    _audit_cache["result"] = result


@router.get("/{job_id}")
def get_report(job_id: str):
    """Generate a PDF audit report for a completed job and stream it."""
    from src.api.routers.audit import _jobs
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="No completed audit found for this job_id")

    result_data = job["result"]

    try:
        from src.utils.report_generator import generate_pdf_bytes
        pdf_bytes = generate_pdf_bytes(result_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=aequitas_audit_{job_id}.pdf"},
    )
