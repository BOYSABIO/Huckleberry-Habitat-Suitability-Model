"""
Inference package — coordinate-based workflow.

Uses model.predictor.HabitatPredictor internally after feature extraction.
"""

from src.inference.pipeline import InferencePipeline

__all__ = ["InferencePipeline"]
