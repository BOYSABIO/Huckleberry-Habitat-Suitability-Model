"""Deprecated module — use src.model.artifact_registry instead."""

from src.model.artifact_registry import (
    ModelArtifactRegistry,
    ModelRegistry,
    load_current_model,
    register_model,
)

__all__ = ["ModelRegistry", "ModelArtifactRegistry", "register_model", "load_current_model"]
