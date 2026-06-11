"""
Model type registry — maps names to trainable implementations.

To add a new model type, register its class here.
"""

from typing import Any, Dict, Type

from src.model.implementations.ensemble import EnsembleModel
from src.model.implementations.random_forest import RandomForestModel

ModelClass = Type[Any]

MODEL_REGISTRY: Dict[str, ModelClass] = {
    "random_forest": RandomForestModel,
    "ensemble": EnsembleModel,
}


def get_model_class(model_type: str) -> ModelClass:
    """Return the implementation class for a model type name."""
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model type '{model_type}'. Supported: {supported}")
    return MODEL_REGISTRY[model_type]


def create_model(model_type: str, **kwargs: Any) -> Any:
    """Instantiate a registered model implementation."""
    return get_model_class(model_type)(**kwargs)
