#!/usr/bin/env python3
"""Basic tests for the Huckleberry Habitat Prediction System."""

import unittest

import numpy as np
import pandas as pd

from src.model.implementations.ensemble import EnsembleModel
from src.model.implementations.random_forest import RandomForestModel
from src.model.registry import MODEL_REGISTRY, create_model


class TestModelRegistry(unittest.TestCase):
    def test_registered_model_types(self):
        self.assertIn("random_forest", MODEL_REGISTRY)
        model = create_model("random_forest")
        self.assertIsInstance(model, RandomForestModel)


class TestRandomForestModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.test_data = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "feature_3": np.random.randn(100),
                "occurrence": np.random.randint(0, 2, 100),
            }
        )

    def test_model_initialization(self):
        model = RandomForestModel()
        self.assertFalse(model.is_fitted)

    def test_training_feature_preparation(self):
        model = RandomForestModel()
        metrics = model.fit(self.test_data, target_col="occurrence", test_size=0.2)
        self.assertIn("accuracy", metrics)
        self.assertEqual(len(model.feature_names), 3)


class TestEnsembleModel(unittest.TestCase):
    def test_model_initialization(self):
        try:
            model = EnsembleModel()
            self.assertIsNotNone(model)
        except ImportError:
            self.skipTest("TPOT/XGBoost not installed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
