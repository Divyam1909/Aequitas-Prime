"""
Phase 16.1 — Pre-bake Demo State
Runs the full pipeline on Adult Income and saves:
  - models/baseline_demo.joblib + baseline_demo_meta.json
  - models/audit_result_demo.pkl  (cached AuditResult)

Run once before demo:
  python scripts/prebake_demo.py

The Streamlit app checks for audit_result_demo.pkl on startup and loads it
instantly, skipping the ~2-minute pipeline retraining during live demo.
"""

from __future__ import annotations
import pickle
import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_csv
from src.utils.adult_preprocessor import ADULT_CONFIG
from src.ml_pipeline.pipeline import run_full_pipeline
from src.ml_pipeline.model_store import save_artifact

DATA_PATH  = Path("data/raw/adult.csv")
DEMO_PKL   = Path("models/audit_result_demo.pkl")
MODELS_DIR = Path("models")


def main():
    print("=" * 60)
    print("  Aequitas Prime — Pre-baking Demo State")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATA_PATH}")
        print("  Run: python scripts/download_data.py")
        sys.exit(1)

    MODELS_DIR.mkdir(exist_ok=True)

    print(f"\n[1/3] Loading dataset from {DATA_PATH} ...")
    df = load_csv(str(DATA_PATH))
    print(f"      Loaded {len(df):,} rows x {len(df.columns)} columns")

    print("\n[2/3] Running full pipeline (baseline + reweighing + ExponentiatedGradient) ...")
    print("      This takes 2-4 minutes. Coffee time.")
    result = run_full_pipeline(df, ADULT_CONFIG, run_inprocessing=True)

    print("\n[3/3] Saving artifacts ...")

    # Save the baseline model for the inference endpoint
    if result.baseline_train and result.baseline_train.model is not None:
        acc = result.baseline_eval.performance.accuracy if result.baseline_eval else None
        save_artifact(
            result.baseline_train.model,
            model_type="baseline",
            config=result.config,
            metrics={"accuracy": acc},
            feature_names=result.baseline_train.feature_names,
            name="baseline_demo",
        )
        print("      Saved models/baseline_demo.joblib + baseline_demo_meta.json")

    # Pickle the full AuditResult for instant UI reload
    with open(DEMO_PKL, "wb") as f:
        pickle.dump(result, f, protocol=4)
    print(f"      Saved {DEMO_PKL} ({DEMO_PKL.stat().st_size / 1024:.0f} KB)")

    # Print key metrics summary
    print("\n" + "=" * 60)
    print("  DEMO STATE READY — Key Metrics Summary")
    print("=" * 60)
    primary = result.config.primary_protected_attr()
    if result.baseline_eval:
        b = result.baseline_eval
        bm = b.fairness.get(primary)
        if bm:
            print(f"  Baseline  : Accuracy={b.performance.accuracy:.1%}  DI={bm.disparate_impact:.3f}  [{bm.severity}]")
        else:
            print(f"  Baseline  : Accuracy={b.performance.accuracy:.1%}")
    if result.preprocessed_eval:
        p = result.preprocessed_eval
        pm = p.fairness.get(primary)
        if pm:
            print(f"  Pre-proc  : Accuracy={p.performance.accuracy:.1%}  DI={pm.disparate_impact:.3f}  [{pm.severity}]")
        else:
            print(f"  Pre-proc  : Accuracy={p.performance.accuracy:.1%}")
    if result.inprocessed_eval:
        i = result.inprocessed_eval
        im = i.fairness.get(primary)
        if im:
            print(f"  In-proc   : Accuracy={i.performance.accuracy:.1%}  DI={im.disparate_impact:.3f}  [{im.severity}]")
        else:
            print(f"  In-proc   : Accuracy={i.performance.accuracy:.1%}")

    proxies = result.proxy_scan or {}
    total_proxies = sum(len(v) for v in proxies.values())
    print(f"\n  Proxy scan: {total_proxies} flagged proxies across all protected attrs")
    print("\n  Demo pre-bake complete. Run: streamlit run ui/app.py")


if __name__ == "__main__":
    main()
