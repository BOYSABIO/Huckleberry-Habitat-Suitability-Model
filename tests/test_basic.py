#!/usr/bin/env python3
"""
Basic tests for the Huckleberry Habitat Prediction System.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Import our modules
try:
    from src.models.pipeline import HuckleberryPredictor
    from src.models.feature_importance import FeatureAnalyzer
except ImportError:
    print("Warning: Could not import modules. Make sure they are properly installed.")


class TestHuckleberryPredictor(unittest.TestCase):
    """Test cases for the main predictor."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        self.test_data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'feature_5': np.random.randn(n_samples),
            'occurrence': np.random.randint(0, 2, n_samples)
        })
    
    def test_predictor_initialization(self):
        """Test that predictor can be initialized."""
        try:
            predictor = HuckleberryPredictor()
            self.assertIsNotNone(predictor)
        except Exception as e:
            self.fail(f"Failed to initialize predictor: {e}")
    
    def test_data_preparation(self):
        """Test data preparation functionality."""
        try:
            predictor = HuckleberryPredictor()
            features, target = predictor.prepare_data(self.test_data)
            
            self.assertEqual(len(features.columns), 5)
            self.assertEqual(len(target), 100)
            self.assertIn('occurrence', self.test_data.columns)
        except Exception as e:
            self.fail(f"Failed to prepare data: {e}")


class TestFeatureAnalyzer(unittest.TestCase):
    """Test cases for feature analyzer."""
    
    def test_analyzer_initialization(self):
        """Test that analyzer can be initialized."""
        try:
            analyzer = FeatureAnalyzer()
            self.assertIsNotNone(analyzer)
        except Exception as e:
            self.fail(f"Failed to initialize analyzer: {e}")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
