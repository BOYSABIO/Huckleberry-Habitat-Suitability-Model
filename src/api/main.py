"""
Main API module for the Huckleberry Habitat API.
"""

from contextlib import asynccontextmanager

import os
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from src.api.schemas import PredictRequest, PredictResponse
from src.model.predictor import load_predictor_for_api

MODEL_PATH = os.getenv("MODEL_PATH")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI app.
    """
    # STARTUP — runs once before accepting requests
    app.state.predictor = load_predictor_for_api(MODEL_PATH)
    yield
    # SHUTDOWN — optional cleanup after yield
    app.state.predictor = None


app = FastAPI(title="Huckleberry Habitat API", lifespan=lifespan)


@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        return {"status": "error", "model_loaded": False}
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": predictor.version_id,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    """
    Predict endpoint.
    """
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    row = body.model_dump()
    features = pd.DataFrame([row])

    result = predictor.predict_with_interval(features)

    if not result.confidence_intervals:
        raise HTTPException(
            status_code=500,
            detail="Confidence interval not available for this model type",
        )

    low, high = result.confidence_intervals[0]

    return PredictResponse(
        probability=float(result.probabilities[0]),
        confidence_interval=[low, high],
    )
