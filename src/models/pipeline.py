"""
Deprecated module — implementations moved to src.model.implementations.

Kept for backward compatibility with notebooks and older imports.
"""

from src.model.implementations.ensemble import EnsembleModel as HuckleberryPredictor
from src.model.implementations.random_forest import RandomForestModel as RandomForestPredictor

__all__ = ["HuckleberryPredictor", "RandomForestPredictor"]
