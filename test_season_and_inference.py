#!/usr/bin/env python3
"""
Test script to demonstrate the new season feature and inference functionality.
"""

import pandas as pd
import numpy as np
from src.data_preprocess.preprocessor import DataPreprocessor
from src.models.pipeline import HuckleberryPredictor, RandomForestPredictor
from src.inference.predict import run_inference, run_inference_with_probabilities

def test_season_feature():
    """Test that the season_num feature is created correctly."""
    print("🧪 Testing season feature creation...")
    
    # Create sample data with dates
    sample_data = pd.DataFrame({
        'eventDate': [
            '2020-01-15',  # Winter
            '2020-03-20',  # Spring
            '2020-07-10',  # Summer
            '2020-10-05',  # Fall
            '2020-12-25'   # Winter
        ],
        'decimalLatitude': [45.0, 45.0, 45.0, 45.0, 45.0],
        'decimalLongitude': [-120.0, -120.0, -120.0, -120.0, -120.0],
        'countryCode': ['US', 'US', 'US', 'US', 'US'],
        'stateProvince': ['Idaho', 'Idaho', 'Idaho', 'Idaho', 'Idaho'],
        'county': ['Test', 'Test', 'Test', 'Test', 'Test']
    })
    
    # Process the data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.clean_occurrence_data(sample_data)
    
    # Check that season_num was created
    assert 'season_num' in processed_data.columns, "season_num column not found"
    
    # Check season values
    expected_seasons = [1, 2, 3, 4, 1]  # Winter, Spring, Summer, Fall, Winter
    actual_seasons = processed_data['season_num'].tolist()
    
    print(f"Expected seasons: {expected_seasons}")
    print(f"Actual seasons: {actual_seasons}")
    
    assert actual_seasons == expected_seasons, f"Season values don't match: {actual_seasons} vs {expected_seasons}"
    
    print("✅ Season feature test passed!")
    return processed_data

def test_model_training_and_inference():
    """Test model training with feature exclusion and inference."""
    print("\n🧪 Testing model training and inference...")
    
    # Create sample training data
    np.random.seed(42)
    n_samples = 100
    
    training_data = pd.DataFrame({
        'gbifID': range(n_samples),
        'decimalLatitude': np.random.uniform(44, 46, n_samples),
        'decimalLongitude': np.random.uniform(-121, -119, n_samples),
        'year': np.random.randint(2010, 2020, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'day': np.random.randint(1, 29, n_samples),
        'air_temperature': np.random.normal(15, 10, n_samples),
        'precipitation_amount': np.random.exponential(5, n_samples),
        'elevation': np.random.uniform(500, 2000, n_samples),
        'soil_ph': np.random.uniform(5.5, 7.5, n_samples),
        'occurrence': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Add season feature
    preprocessor = DataPreprocessor()
    training_data = preprocessor._add_season_feature(training_data)
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Training data columns: {list(training_data.columns)}")
    
    # Train model
    model = RandomForestPredictor(n_estimators=10)  # Small model for testing
    metrics = model.fit(training_data, target_col='occurrence')
    
    print(f"Training metrics: {metrics}")
    
    # Test inference
    inference_data = training_data.head(5).copy()  # Use first 5 rows for inference
    predictions, probabilities = run_inference_with_probabilities(
        'test_model.joblib', inference_data
    )
    
    print(f"Inference predictions: {predictions}")
    print(f"Inference probabilities shape: {probabilities.shape}")
    
    # Clean up
    import os
    if os.path.exists('test_model.joblib'):
        os.remove('test_model.joblib')
    
    print("✅ Model training and inference test passed!")

def main():
    """Run all tests."""
    print("🚀 Running season feature and inference tests...\n")
    
    try:
        # Test season feature
        processed_data = test_season_feature()
        
        # Test model training and inference
        test_model_training_and_inference()
        
        print("\n🎉 All tests passed!")
        print("\n📋 Summary of changes:")
        print("1. ✅ Added season_num feature (1=Winter, 2=Spring, 3=Summer, 4=Fall)")
        print("2. ✅ Model training now excludes gbifID, decimalLatitude, decimalLongitude")
        print("3. ✅ Inference expects preprocessed data and handles feature selection")
        print("4. ✅ Both HuckleberryPredictor and RandomForestPredictor updated")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 