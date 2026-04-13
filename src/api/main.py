"""
Aequitas Prime — FastAPI application entry point.
"""

from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import audit, inference, report


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pre-trained model on startup if one exists."""
    try:
        from src.ml_pipeline.model_store import list_artifacts, load_artifact
        from src.utils.adult_preprocessor import ADULT_CONFIG
        from src.api.routers.inference import set_model

        artifacts = list_artifacts()
        if artifacts:
            # Load the most recent baseline model
            name = sorted(artifacts)[-1]
            model, meta = load_artifact(name)
            feature_names = meta.get("feature_names", [])
            set_model(model, feature_names, ADULT_CONFIG)
            print(f"  [startup] Loaded model: {name}")
        else:
            print("  [startup] No saved model found. Run the pipeline first.")
    except Exception as e:
        print(f"  [startup] Model load skipped: {e}")
    yield


app = FastAPI(
    title="Aequitas Prime",
    description="Real-Time Algorithmic Fairness & Bias Neutralization Suite",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit (localhost:8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audit.router)
app.include_router(inference.router)
app.include_router(report.router)


@app.get("/")
def root():
    return {
        "name": "Aequitas Prime",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
