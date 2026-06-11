"""
Inference package.

- model.artifact.HabitatPredictor: score a prepared feature matrix (API path)
- inference.pipeline.InferencePipeline: coordinates → features → score (CLI path)
"""

from src.inference.pipeline import InferencePipeline

__all__ = ["InferencePipeline"]
