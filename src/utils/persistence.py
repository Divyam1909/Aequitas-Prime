"""
Audit Persistence — save and load bias audit results to/from a local JSON store.

Each audit is stored as a JSON file in an `audits/` directory under the project
root (or a user-configurable path).  A lightweight index file `audits/index.json`
tracks all saved audits so the history sidebar can load them without scanning
every file.

Stored per audit:
  - Unique audit ID  (timestamp + dataset slug)
  - Dataset name + row/feature counts
  - Task type
  - Primary protected attribute
  - Key fairness metrics (baseline DI, best DI, severity)
  - Model accuracy (baseline and best)
  - Proxy count
  - Timestamp (ISO 8601)
  - Full metrics dict (serialisable subset)

Design decisions:
  - No external DB dependency — plain JSON files work on Streamlit Cloud.
  - AuditResult is NOT fully serialisable (contains numpy arrays, sklearn
    models, etc.).  We store only the metrics/metadata that are needed for the
    history table and comparison chart.
  - Loading returns AuditSummary objects (lightweight), not AuditResult.
"""

from __future__ import annotations
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Default storage directory ─────────────────────────────────────────────────
_DEFAULT_AUDIT_DIR = Path(__file__).resolve().parent.parent.parent / "audits"


@dataclass
class AuditSummary:
    """Lightweight, fully serialisable snapshot of one audit run."""
    audit_id: str
    dataset_name: str
    timestamp: str                    # ISO 8601
    task_type: str
    n_rows: int
    n_features: int
    primary_attr: str

    # Baseline
    baseline_accuracy: float
    baseline_di: float | None         # None for regression
    baseline_severity: str

    # Best mitigated
    best_model: str
    best_accuracy: float
    best_di: float | None
    best_severity: str

    # Extras
    proxy_count: int
    all_metrics: dict[str, Any] = field(default_factory=dict)


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to plain Python for json.dumps."""
    import math
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    return obj


def _extract_summary(result: Any, audit_id: str) -> AuditSummary:
    """
    Extract a serialisable AuditSummary from an AuditResult.

    Uses duck-typing so we don't import AuditResult here (avoids circular deps).
    """
    import math

    config = result.config
    primary = config.primary_protected_attr()
    task    = getattr(config, "task_type", "binary")

    # Baseline metrics
    b_eval = result.baseline_eval
    b_acc  = float(b_eval.performance.accuracy) if b_eval else float("nan")
    b_di: float | None = None
    b_sev  = "UNKNOWN"
    if b_eval and b_eval.fairness:
        bm = b_eval.fairness.get(primary)
        if bm:
            b_di  = None if (bm.disparate_impact is None or math.isnan(bm.disparate_impact)) else float(bm.disparate_impact)
            b_sev = bm.severity or "UNKNOWN"

    # Best mitigated eval (inprocessed > preprocessed > baseline)
    best_eval = result.inprocessed_eval or result.preprocessed_eval or result.baseline_eval
    best_model_name = best_eval.model_type if best_eval else "baseline"
    best_acc  = float(best_eval.performance.accuracy) if best_eval else float("nan")
    best_di: float | None = None
    best_sev  = "UNKNOWN"
    if best_eval and best_eval.fairness:
        bm2 = best_eval.fairness.get(primary)
        if bm2:
            best_di  = None if (bm2.disparate_impact is None or math.isnan(bm2.disparate_impact)) else float(bm2.disparate_impact)
            best_sev = bm2.severity or "UNKNOWN"

    # Proxy count
    proxy_count = sum(len(v) for v in result.proxy_scan.values()) if result.proxy_scan else 0

    # Condensed all_metrics for history chart
    all_metrics: dict[str, Any] = {}
    for model_tag, ev in [
        ("baseline", result.baseline_eval),
        ("preprocessed", result.preprocessed_eval),
        ("inprocessed", result.inprocessed_eval),
    ]:
        if ev is None:
            continue
        m = ev.fairness.get(primary)
        all_metrics[model_tag] = {
            "accuracy":          _to_json_safe(ev.performance.accuracy),
            "f1":                _to_json_safe(ev.performance.f1),
            "disparate_impact":  _to_json_safe(m.disparate_impact if m else None),
            "stat_parity_diff":  _to_json_safe(m.statistical_parity_diff if m else None),
            "eod":               _to_json_safe(m.equalized_odds_diff if m else None),
            "severity":          m.severity if m else "UNKNOWN",
        }

    return AuditSummary(
        audit_id=audit_id,
        dataset_name=result.dataset_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        task_type=task,
        n_rows=result.n_rows,
        n_features=result.n_features,
        primary_attr=primary,
        baseline_accuracy=_to_json_safe(b_acc),
        baseline_di=_to_json_safe(b_di),
        baseline_severity=b_sev,
        best_model=best_model_name,
        best_accuracy=_to_json_safe(best_acc),
        best_di=_to_json_safe(best_di),
        best_severity=best_sev,
        proxy_count=proxy_count,
        all_metrics=all_metrics,
    )


# ── Index management ──────────────────────────────────────────────────────────

def _index_path(audit_dir: Path) -> Path:
    return audit_dir / "index.json"


def _load_index(audit_dir: Path) -> list[dict]:
    idx_path = _index_path(audit_dir)
    if not idx_path.exists():
        return []
    try:
        with open(idx_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_index(audit_dir: Path, index: list[dict]) -> None:
    idx_path = _index_path(audit_dir)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def save_audit(
    result: Any,
    audit_dir: str | Path | None = None,
    audit_id: str | None = None,
) -> str:
    """
    Persist an AuditResult snapshot.

    Parameters
    ----------
    result    : AuditResult from run_full_pipeline()
    audit_dir : directory to write audits (default: <project_root>/audits/)
    audit_id  : custom ID; auto-generated if None

    Returns
    -------
    audit_id string
    """
    dir_ = Path(audit_dir) if audit_dir else _DEFAULT_AUDIT_DIR
    dir_.mkdir(parents=True, exist_ok=True)

    if audit_id is None:
        slug = result.dataset_name.lower().replace(" ", "_")[:20]
        ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        audit_id = f"{ts}_{slug}_{uuid.uuid4().hex[:6]}"

    summary = _extract_summary(result, audit_id)
    summary_dict = _to_json_safe(asdict(summary))

    # Write individual audit file
    audit_path = dir_ / f"{audit_id}.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # Update index
    index = _load_index(dir_)
    # Remove existing entry with same id (overwrite scenario)
    index = [e for e in index if e.get("audit_id") != audit_id]
    index.append({
        "audit_id":      audit_id,
        "dataset_name":  summary.dataset_name,
        "timestamp":     summary.timestamp,
        "task_type":     summary.task_type,
        "primary_attr":  summary.primary_attr,
        "baseline_di":   summary.baseline_di,
        "best_di":       summary.best_di,
        "best_severity": summary.best_severity,
        "n_rows":        summary.n_rows,
    })
    # Keep index sorted newest-first
    index.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    _save_index(dir_, index)

    return audit_id


def list_audits(audit_dir: str | Path | None = None) -> list[dict]:
    """
    Return the audit index as a list of lightweight dicts (newest first).
    Each dict contains: audit_id, dataset_name, timestamp, task_type,
    primary_attr, baseline_di, best_di, best_severity, n_rows.
    """
    dir_ = Path(audit_dir) if audit_dir else _DEFAULT_AUDIT_DIR
    return _load_index(dir_)


def load_audit(audit_id: str, audit_dir: str | Path | None = None) -> AuditSummary | None:
    """
    Load a full AuditSummary by ID.  Returns None if not found.
    """
    dir_ = Path(audit_dir) if audit_dir else _DEFAULT_AUDIT_DIR
    audit_path = dir_ / f"{audit_id}.json"
    if not audit_path.exists():
        return None
    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AuditSummary(**data)
    except (json.JSONDecodeError, TypeError, OSError):
        return None


def delete_audit(audit_id: str, audit_dir: str | Path | None = None) -> bool:
    """
    Delete an audit by ID.  Returns True if deleted, False if not found.
    """
    dir_ = Path(audit_dir) if audit_dir else _DEFAULT_AUDIT_DIR
    audit_path = dir_ / f"{audit_id}.json"
    deleted = False
    if audit_path.exists():
        audit_path.unlink()
        deleted = True
    # Remove from index regardless
    index = _load_index(dir_)
    new_index = [e for e in index if e.get("audit_id") != audit_id]
    if len(new_index) != len(index):
        _save_index(dir_, new_index)
        deleted = True
    return deleted


def audit_history_dataframe(audit_dir: str | Path | None = None):
    """
    Return a pandas DataFrame suitable for the history table in the UI.
    Columns: Timestamp, Dataset, Task, Attr, Baseline DI, Best DI, Improvement, Severity, Rows
    """
    import pandas as pd
    import math
    index = list_audits(audit_dir)
    if not index:
        return pd.DataFrame()

    rows = []
    for e in index:
        b_di   = e.get("baseline_di")
        bt_di  = e.get("best_di")
        improv = None
        if b_di is not None and bt_di is not None:
            try:
                improv = round(float(bt_di) - float(b_di), 3)
            except (TypeError, ValueError):
                improv = None

        # Format timestamp
        ts = e.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
            ts_fmt = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            ts_fmt = ts[:16] if ts else "—"

        rows.append({
            "Timestamp":   ts_fmt,
            "Dataset":     e.get("dataset_name", "—"),
            "Task":        e.get("task_type", "—"),
            "Attr":        e.get("primary_attr", "—"),
            "Baseline DI": round(float(b_di), 3) if b_di is not None else None,
            "Best DI":     round(float(bt_di), 3) if bt_di is not None else None,
            "ΔDI":         improv,
            "Severity":    e.get("best_severity", "—"),
            "Rows":        e.get("n_rows", "—"),
            "audit_id":    e.get("audit_id", ""),
        })

    return pd.DataFrame(rows)
