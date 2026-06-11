"""
Shared feature-preparation utilities for model training and inference.
"""

from typing import List, Tuple

import pandas as pd

# Columns excluded from the modeling feature matrix
EXCLUDE_COLS = [
    "gbifID",
    "gridmet_lat",
    "gridmet_lon",
    "gridmet_date",
    "decimalLatitude",
    "decimalLongitude",
    "datetime",
    "parsed_datetime",
    "season",
    "month",
    "day",
]


def prepare_training_features(
    data: pd.DataFrame,
    target_col: str = "occurrence",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Split a training dataframe into features, target, and feature names."""
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    drop_cols = [target_col] + [col for col in EXCLUDE_COLS if col in data.columns]
    features = data.drop(drop_cols, axis=1)
    target = data[target_col]
    feature_names = features.columns.tolist()
    return features, target, feature_names


def select_inference_features(
    data: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    """Select and order inference features to match training."""
    inference_data = data.drop(
        [col for col in EXCLUDE_COLS if col in data.columns],
        axis=1,
    )
    missing = set(feature_names) - set(inference_data.columns)
    if missing:
        raise ValueError(f"Missing required features for inference: {missing}")
    return inference_data[feature_names]
