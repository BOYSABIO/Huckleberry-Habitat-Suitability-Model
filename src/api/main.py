import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.model.predictor import load_predictor_from_path

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/huckleberry_model_v13_20260612_111519.joblib",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP — runs once before accepting requests
    app.state.predictor = load_predictor_from_path(MODEL_PATH)
    yield
    # SHUTDOWN — optional cleanup after yield
    app.state.predictor = None


app = FastAPI(title="Huckleberry Habitat API", lifespan=lifespan)


@app.get("/health")
def health():
    predictor = app.state.predictor
    if predictor is None:
        return {"status": "error", "model_loaded": False}
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": predictor.version_id,
    }