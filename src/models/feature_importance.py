"""Deprecated module — use src.evaluation.feature_importance instead."""

from src.evaluation.feature_importance import (
    extract_feature_importance,
    generate_feature_importance_outputs,
    save_training_outputs,
)

__all__ = [
    "extract_feature_importance",
    "generate_feature_importance_outputs",
    "save_training_outputs",
]
