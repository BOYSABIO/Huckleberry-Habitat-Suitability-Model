"""
Training entry point.

Picks a registered model type, runs its fit() (split / scale / train / metrics),
and returns the fitted instance. Saving to disk is handled by training/pipeline.py
via model/store.py.
"""

from typing import Any, Dict, Tuple

import pandas as pd

import src.model.implementations  # noqa: F401 — register model types
from src.config.settings import ModelSettings
from src.model.registry import create_model


def train_model(
    df: pd.DataFrame,
    settings: ModelSettings,
) -> Tuple[Any, Dict[str, float]]:
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
