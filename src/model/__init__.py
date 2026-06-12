"""Model package — registry, training, persistence, and scoring."""

import src.model.implementations  # noqa: F401

from src.model.predictor import HabitatPredictor, load_predictor_from_path, load_predictor_from_settings
from src.model.registry import MODEL_REGISTRY, create_model, get_model_class, register
from src.model.store import ModelArtifactRegistry, load_current_model, register_model
from src.model.trainer import train_model

__all__ = [
    "HabitatPredictor",
    "MODEL_REGISTRY",
    "ModelArtifactRegistry",
    "create_model",
    "get_model_class",
    "load_current_model",
    "load_predictor_from_path",
    "load_predictor_from_settings",
    "register",
    "register_model",
    "train_model",
]
