#!/usr/bin/env python3
"""
Test script for model creation and training.
This script tests both Random Forest and Ensemble models on existing enriched data
and synthetic data to ensure the pipeline works correctly.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import random

# Add src to path
sys.path.append('src')

from models.pipeline import RandomForestPredictor, HuckleberryPredictor
from features.environmental import EnvironmentalDataExtractor
from data_preprocess.preprocessor import DataPreprocessor

def create_synthetic_data(n_samples=1000):
    """
    Create synthetic data that matches the exact format of our enriched dataset.
    This ensures we test with the same column structure as real data.
    """
    print("Creating synthetic data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create base data
    data = []
    
    for i in range(n_samples):
        # Random coordinates within reasonable bounds
        lat = np.random.uniform(40, 49)  # Roughly Idaho/Washington area
        lon = np.random.uniform(-125, -104)
        
        # Random date between 1979-2020
        start_date = datetime(1979, 1, 1)
        end_date = datetime(2020, 12, 31)
        random_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        
        # Random occurrence (1 for presence, 0 for absence)
        occurrence = random.choice([0, 1])
        
        # Environmental variables (realistic ranges based on our data)
        air_temp = np.random.uniform(250, 310)  # Kelvin
        precip = np.random.uniform(0, 20)
        humidity = np.random.uniform(0.001, 0.02)
        rel_humidity = np.random.uniform(10, 100)
        vpd = np.random.uniform(0, 3)
        pet = np.random.uniform(0, 12)
        solar_flux = np.random.uniform(50, 400)
        wind_speed = np.random.uniform(1, 10)
        elevation = np.random.uniform(100, 3000)
        soil_ph = np.random.uniform(4.5, 8.5)
        
        # Extract date components
        day = random_date.day
        month = random_date.month
        year = random_date.year
        
        # Calculate season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        if month in [12, 1, 2]:
            season_num = 1
        elif month in [3, 4, 5]:
            season_num = 2
        elif month in [6, 7, 8]:
            season_num = 3
        else:
            season_num = 4
        
        # Create row with exact same format as enriched data
        row = {
            'gbifID': f"synthetic_{i}" if occurrence == 1 else None,
            'datetime': random_date.strftime('%Y-%m-%d %H:%M:%S') if occurrence == 1 else random_date.strftime('%Y-%m-%d'),
            'decimalLatitude': lat,
            'decimalLongitude': lon,
            'gridmet_lat': round(lat, 6),
            'gridmet_lon': round(lon, 6),
            'gridmet_date': random_date.strftime('%Y-%m-%d %H:%M:%S') if occurrence == 1 else random_date.strftime('%Y-%m-%d'),
            'air_temperature': air_temp,
            'precipitation_amount': precip,
            'specific_humidity': humidity,
            'relative_humidity': rel_humidity,
            'mean_vapor_pressure_deficit': vpd,
            'potential_evapotranspiration': pet,
            'surface_downwelling_shortwave_flux_in_air': solar_flux,
            'wind_speed': wind_speed,
            'occurrence': occurrence,
            'elevation': elevation,
            'soil_ph': soil_ph,
            'day': day,
            'month': month,
            'year': year,
            'season_num': season_num
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    print(f"Created synthetic data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Occurrence distribution: {df['occurrence'].value_counts().to_dict()}")
    
    return df

def test_random_forest_model():
    """Test Random Forest model creation and training."""
    print("\n" + "="*50)
    print("TESTING RANDOM FOREST MODEL")
    print("="*50)
    
    # Load existing enriched data from our pipeline output
    enriched_data_path = "data/enriched/huckleberry_final_enriched.csv"
    
    if os.path.exists(enriched_data_path):
        print(f"Loading existing enriched data from {enriched_data_path}")
        df = pd.read_csv(enriched_data_path)
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Occurrence distribution: {df['occurrence'].value_counts().to_dict()}")
        
        # Add missing season_num column
        if 'season_num' not in df.columns:
            print("Adding missing season_num column...")
            df['season_num'] = df['month'].apply(lambda m: 1 if m in [12, 1, 2] else 
                                                          2 if m in [3, 4, 5] else 
                                                          3 if m in [6, 7, 8] else 4)
        
        # Handle missing values
        print("Handling missing values...")
        df = df.dropna()  # Remove rows with any missing values
        print(f"Data shape after removing missing values: {df.shape}")
        
    else:
        print("No existing enriched data found, using synthetic data")
        df = create_synthetic_data(1000)
    
    # Initialize Random Forest predictor
    rf_predictor = RandomForestPredictor(n_estimators=100, random_state=42)
    
    # Train the model
    print("\nTraining Random Forest model...")
    metrics = rf_predictor.fit(df, target_col='occurrence', test_size=0.2, random_state=42)
    
    print(f"Training completed!")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Train size: {metrics['train_size']}")
    print(f"Test size: {metrics['test_size']}")
    
    # Test feature importance
    print("\nFeature importance:")
    importance_df = rf_predictor.get_feature_importance()
    print(importance_df.head(10))
    
    # Test prediction on a small sample
    print("\nTesting predictions on a small sample...")
    sample_data = df.head(10)
    predictions = rf_predictor.predict(sample_data)
    probabilities = rf_predictor.model.predict_proba(rf_predictor.scaler.transform(
        rf_predictor.prepare_inference_data(sample_data)
    ))
    
    print("Sample predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        if prob.ndim == 1 and len(prob) == 1:
            # Single probability case (likely for small datasets)
            print(f"Sample {i+1}: Prediction={pred}, Probability=[{prob[0]:.3f}]")
        else:
            # Standard binary classification case
            print(f"Sample {i+1}: Prediction={pred}, Probability=[{prob[0]:.3f}, {prob[1]:.3f}]")
    
    # Save the model as test_random_forest
    model_path = "models/test_random_forest.joblib"
    print(f"\nSaving model to {model_path}")
    rf_predictor.save_model(model_path)
    
    # Test loading the model
    print("Testing model loading...")
    loaded_predictor = RandomForestPredictor()
    loaded_predictor.load_model(model_path)
    
    # Verify loaded model works
    test_predictions = loaded_predictor.predict(sample_data)
    print(f"Loaded model predictions match: {np.array_equal(predictions, test_predictions)}")
    
    return rf_predictor, metrics

def test_ensemble_model():
    """Test Ensemble model creation and training."""
    print("\n" + "="*50)
    print("TESTING ENSEMBLE MODEL")
    print("="*50)
    
    # Load existing enriched data from our pipeline output
    enriched_data_path = "data/enriched/huckleberry_final_enriched.csv"
    
    if os.path.exists(enriched_data_path):
        print(f"Loading existing enriched data from {enriched_data_path}")
        df = pd.read_csv(enriched_data_path)
        print(f"Loaded data shape: {df.shape}")
        
        # Add missing season_num column
        if 'season_num' not in df.columns:
            print("Adding missing season_num column...")
            df['season_num'] = df['month'].apply(lambda m: 1 if m in [12, 1, 2] else 
                                                          2 if m in [3, 4, 5] else 
                                                          3 if m in [6, 7, 8] else 4)
        
        # Handle missing values
        print("Handling missing values...")
        df = df.dropna()  # Remove rows with any missing values
        print(f"Data shape after removing missing values: {df.shape}")
        
    else:
        print("No existing enriched data found, using synthetic data")
        df = create_synthetic_data(1000)
    
    # Initialize Ensemble predictor
    ensemble_predictor = HuckleberryPredictor()
    
    # Train the model
    print("\nTraining Ensemble model...")
    metrics = ensemble_predictor.fit(df, target_col='occurrence', test_size=0.2, random_state=42)
    
    print(f"Training completed!")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Train size: {metrics['train_size']}")
    print(f"Test size: {metrics['test_size']}")
    
    # Test feature importance
    print("\nFeature importance:")
    importance_df = ensemble_predictor.get_feature_importance()
    print(importance_df.head(10))
    
    # Test prediction on a small sample
    print("\nTesting predictions on a small sample...")
    sample_data = df.head(10)
    predictions = ensemble_predictor.predict(sample_data)
    probabilities = ensemble_predictor.predict_proba(sample_data)
    
    print("Sample predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        if prob.ndim == 1 and len(prob) == 1:
            # Single probability case (likely for small datasets)
            print(f"Sample {i+1}: Prediction={pred}, Probability=[{prob[0]:.3f}]")
        else:
            # Standard binary classification case
            print(f"Sample {i+1}: Prediction={pred}, Probability=[{prob[0]:.3f}, {prob[1]:.3f}]")
    
    # Save the model as test_ensemble
    model_path = "models/test_ensemble.joblib"
    print(f"\nSaving model to {model_path}")
    ensemble_predictor.save_model(model_path)
    
    # Test loading the model
    print("Testing model loading...")
    loaded_predictor = HuckleberryPredictor()
    loaded_predictor.load_model(model_path)
    
    # Verify loaded model works
    test_predictions = loaded_predictor.predict(sample_data)
    print(f"Loaded model predictions match: {np.array_equal(predictions, test_predictions)}")
    
    return ensemble_predictor, metrics

def test_data_format_consistency():
    """Test that our data format is consistent with what the model expects."""
    print("\n" + "="*50)
    print("TESTING DATA FORMAT CONSISTENCY")
    print("="*50)
    
    # Load existing enriched data from our pipeline output
    enriched_data_path = "data/enriched/huckleberry_final_enriched.csv"
    
    if os.path.exists(enriched_data_path):
        df = pd.read_csv(enriched_data_path)
        print(f"Loaded enriched data shape: {df.shape}")
        
        # Add missing season_num column
        if 'season_num' not in df.columns:
            print("Adding missing season_num column...")
            df['season_num'] = df['month'].apply(lambda m: 1 if m in [12, 1, 2] else 
                                                          2 if m in [3, 4, 5] else 
                                                          3 if m in [6, 7, 8] else 4)
        
        # Handle missing values
        print("Handling missing values...")
        df = df.dropna()  # Remove rows with any missing values
        print(f"Data shape after removing missing values: {df.shape}")
        
        # Check required columns
        required_cols = [
            'gbifID', 'datetime', 'decimalLatitude', 'decimalLongitude',
            'gridmet_lat', 'gridmet_lon', 'gridmet_date',
            'air_temperature', 'precipitation_amount', 'specific_humidity',
            'relative_humidity', 'mean_vapor_pressure_deficit',
            'potential_evapotranspiration', 'surface_downwelling_shortwave_flux_in_air',
            'wind_speed', 'occurrence', 'elevation', 'soil_ph',
            'day', 'month', 'year', 'season_num'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"WARNING: Missing columns: {missing_cols}")
        else:
            print("✓ All required columns present")
        
        # Test data preparation
        rf_predictor = RandomForestPredictor()
        features, target = rf_predictor.prepare_data(df, target_col='occurrence')
        
        print(f"Features shape after preparation: {features.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Feature columns: {features.columns.tolist()}")
        
        # Verify excluded columns are not in features
        excluded_cols = ['gbifID', 'decimalLatitude', 'decimalLongitude']
        excluded_in_features = [col for col in excluded_cols if col in features.columns]
        if excluded_in_features:
            print(f"ERROR: Excluded columns found in features: {excluded_in_features}")
        else:
            print("✓ Excluded columns properly removed from features")
        
        # Verify date components are included
        date_cols = ['day', 'month', 'year', 'season_num']
        missing_date_cols = [col for col in date_cols if col not in features.columns]
        if missing_date_cols:
            print(f"ERROR: Missing date components in features: {missing_date_cols}")
        else:
            print("✓ Date components properly included in features")
        
        return True
    else:
        print("No enriched data found for format testing")
        return False

def main():
    """Run all tests."""
    print("Starting model creation tests...")
    
    # Test data format consistency first
    format_ok = test_data_format_consistency()
    
    if format_ok:
        # Test Random Forest model
        rf_predictor, rf_metrics = test_random_forest_model()
        
        # Test Ensemble model
        ensemble_predictor, ensemble_metrics = test_ensemble_model()
        
        # Compare results
        print("\n" + "="*50)
        print("COMPARISON OF RESULTS")
        print("="*50)
        print(f"Random Forest Accuracy: {rf_metrics['accuracy']:.4f}")
        print(f"Ensemble Accuracy: {ensemble_metrics['accuracy']:.4f}")
        
        if rf_metrics['accuracy'] > ensemble_metrics['accuracy']:
            print("Random Forest performed better!")
        elif ensemble_metrics['accuracy'] > rf_metrics['accuracy']:
            print("Ensemble model performed better!")
        else:
            print("Both models performed equally!")
        
        print("\nAll tests completed successfully!")
        print("Models saved as:")
        print("- models/test_random_forest.joblib")
        print("- models/test_ensemble.joblib")
    else:
        print("Data format test failed. Please check your enriched data.")

if __name__ == "__main__":
    main() 