"""
Training entry point for registered model types.
"""

from typing import Any, Dict, Tuple

import pandas as pd

from src.config.settings import ModelSettings
from src.model.registry import create_model


def train_model(
    df: pd.DataFrame,
    settings: ModelSettings,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a model using the type specified in settings.

    Returns:
        Tuple of (fitted model instance, metrics dict)
    """
    kwargs = {}
    if settings.model_type == "random_forest":
        kwargs = {
            "n_estimators": settings.n_estimators,
            "random_state": settings.random_state,
        }

    model = create_model(settings.model_type, **kwargs)
    metrics = model.fit(
        df,
        target_col=settings.target_column,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    return model, metrics
