"""
Model package — training implementations, artifact storage, and scoring.
"""

from src.model.artifact import HabitatPredictor, load_predictor_from_path, load_predictor_from_settings
from src.model.artifact_registry import ModelArtifactRegistry, ModelRegistry
from src.model.registry import create_model, get_model_class
from src.model.trainer import train_model

__all__ = [
    "HabitatPredictor",
    "ModelArtifactRegistry",
    "ModelRegistry",
    "create_model",
    "get_model_class",
    "load_predictor_from_path",
    "load_predictor_from_settings",
    "train_model",
]
