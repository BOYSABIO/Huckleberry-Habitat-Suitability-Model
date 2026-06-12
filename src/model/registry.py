"""
Model type registry.

Register implementations with @register("name") and instantiate via create_model().
"""

from typing import Any, Callable, Dict, Type

ModelClass = Type[Any]

MODEL_REGISTRY: Dict[str, ModelClass] = {}


def register(name: str) -> Callable[[ModelClass], ModelClass]:
    """Decorator to add a model class to the type catalog."""

    def decorator(cls: ModelClass) -> ModelClass:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(model_type: str) -> ModelClass:
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model type '{model_type}'. Supported: {supported}")
    return MODEL_REGISTRY[model_type]


def create_model(model_type: str, **kwargs: Any) -> Any:
    return get_model_class(model_type)(**kwargs)
