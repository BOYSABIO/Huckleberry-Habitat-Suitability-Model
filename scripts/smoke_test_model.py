#!/usr/bin/env python3
"""Load the production joblib artifact and run a single prediction."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.artifact import load_predictor_from_path

MODEL_PATH = "models/random_forest_improved.joblib"


def main() -> None:
    predictor = load_predictor_from_path(MODEL_PATH)
    print(f"Loaded model: {predictor.version_id}")
    print(f"Features ({len(predictor.feature_names)}): {predictor.feature_names}")

    row = {name: 0.0 for name in predictor.feature_names}
    features = pd.DataFrame([row])
    result = predictor.predict_with_interval(features)

    print(f"probability: {result.probabilities[0]:.4f}")
    if result.confidence_intervals:
        low, high = result.confidence_intervals[0]
        print(f"confidence_interval: [{low:.4f}, {high:.4f}]")


if __name__ == "__main__":
    main()
