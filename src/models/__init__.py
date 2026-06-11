"""
Deprecated package — prefer src.model, src.inference, and src.evaluation.
"""

from src.model.implementations.ensemble import EnsembleModel as HuckleberryPredictor
from src.evaluation.feature_importance import generate_feature_importance_outputs

__all__ = ["HuckleberryPredictor", "generate_feature_importance_outputs"]
