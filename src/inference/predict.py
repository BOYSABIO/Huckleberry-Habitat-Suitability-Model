"""
Inference module for huckleberry pipeline.
"""

import pandas as pd
import joblib
import logging
from typing import Any

logger = logging.getLogger(__name__)

def run_inference(model_path: str, data: pd.DataFrame) -> Any:
    """
    Run inference using a trained model.
    Args:
        model_path: Path to the trained model file
        data: DataFrame with features for prediction
    Returns:
        Model predictions
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Running inference on {len(data)} samples")
    predictions = model.predict(data)
    return predictions 