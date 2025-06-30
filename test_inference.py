#!/usr/bin/env python3
"""
Test script for the inference functionality using existing trained models.
"""

import os
import pandas as pd
from datetime import datetime

# Import from our main inference module
from src.inference.predict import (
    create_prediction_grid, 
    predict_habitat_suitability, 
    create_prediction_map
)

def test_inference_with_existing_models():
    """Test inference using the models in the models folder."""
    
    print("🧪 Testing Inference with Existing Models")
    print("=" * 50)
    
    # Check available models
    models_dir = "models"
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    print(f"Available models: {available_models}")
    
    if not available_models:
        print("❌ No trained models found in models/ directory")
        return
    
    # Specifically use the random forest model
    model_file = "random_forest_1.joblib"
    if model_file not in available_models:
        print(f"❌ {model_file} not found, using first available model")
        model_file = available_models[0]
    
    model_path = os.path.join(models_dir, model_file)
    print(f"Using model: {model_path}")
    
    # Define a very small test area - just 10 coordinates
    # Using a tiny area to get exactly 10 points (3x3 grid + 1 extra)
    lat_min, lat_max = 44.7, 44.72  # Very small area for testing
    lon_min, lon_max = -116.3, -116.28
    
    print(f"\n📍 Test area: ({lat_min:.3f}, {lon_min:.3f}) to ({lat_max:.3f}, {lon_max:.3f})")
    print(f"Resolution: 0.01 degrees (should give ~10 coordinates)")
    
    # Set target date
    target_date = datetime(2020, 7, 15)
    print(f"Target date: {target_date.strftime('%Y-%m-%d')}")
    
    try:
        # Test 1: Create prediction grid
        print("\n🔍 Test 1: Creating prediction grid...")
        grid_df = create_prediction_grid(lat_min, lat_max, lon_min, lon_max, resolution=0.01)
        print(f"✅ Created grid with {len(grid_df)} points")
        
        # Limit to exactly 10 coordinates for testing
        if len(grid_df) > 10:
            grid_df = grid_df.head(10)
            print(f"📏 Limited to first 10 coordinates for testing")
        
        print(f"Final grid size: {len(grid_df)} points")
        print(f"Grid bounds: {grid_df['decimalLatitude'].min():.3f} to {grid_df['decimalLatitude'].max():.3f} lat")
        print(f"Grid bounds: {grid_df['decimalLongitude'].min():.3f} to {grid_df['decimalLongitude'].max():.3f} lon")
        
        # Test 2: Run habitat suitability prediction
        print("\n🚀 Test 2: Running habitat suitability prediction...")
        results = predict_habitat_suitability(
            model_path=model_path,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            resolution=0.01,
            target_date=target_date,
            output_path="outputs/test_inference_results.csv"
        )
        
        # Limit results to 10 coordinates
        if len(results) > 10:
            results = results.head(10)
            print(f"📏 Limited results to first 10 coordinates")
        
        # Display results
        print("\n📊 Prediction Results:")
        print(f"Total grid points: {len(results)}")
        print(f"Predicted suitable habitat: {(results['prediction'] == 1).sum()}")
        print(f"Predicted unsuitable habitat: {(results['prediction'] == 0).sum()}")
        
        if len(results) > 0:
            print(f"\n📍 All predictions (10 coordinates):")
            for i, (_, row) in enumerate(results.iterrows(), 1):
                status = "SUITABLE" if row['prediction'] == 1 else "UNSUITABLE"
                prob = row['probability_presence'] if row['prediction'] == 1 else row['probability_absence']
                print(f"  {i}. ({row['decimalLatitude']:.3f}, {row['decimalLongitude']:.3f}): "
                      f"{status} (confidence: {prob:.2f})")
            
            # Show environmental features if available
            env_cols = ['elevation', 'soil_ph', 'air_temperature', 'precipitation_amount', 'season_num']
            available_env = [col for col in env_cols if col in results.columns]
            if available_env:
                print(f"\n🌍 Environmental features available: {available_env}")
                for col in available_env:
                    print(f"  {col}: {results[col].min():.2f} to {results[col].max():.2f}")
        
        # Test 3: Create HTML map
        print("\n🗺️ Test 3: Creating HTML map...")
        create_prediction_map(results, "outputs/test_prediction_map.html", confidence_threshold=0.8)
        
        print(f"\n💾 Results saved to: outputs/test_inference_results.csv")
        print("✅ Inference test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during inference test: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have internet connection for environmental data")
        print("2. Check that the model file is not corrupted")
        print("3. Verify the coordinates are within GridMET bounds")
        print("4. Check that all required dependencies are installed")

def test_with_different_models():
    """Test with different available models."""
    
    print("\n" + "="*50)
    print("🧪 Testing with Different Models")
    print("="*50)
    
    models_dir = "models"
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    # Test area - very small for quick testing
    lat_min, lat_max = 44.7, 44.71
    lon_min, lon_max = -116.3, -116.29
    target_date = datetime(2020, 7, 15)
    
    for model_file in available_models:
        model_path = os.path.join(models_dir, model_file)
        print(f"\n🔍 Testing model: {model_file}")
        
        try:
            results = predict_habitat_suitability(
                model_path=model_path,
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                resolution=0.01,
                target_date=target_date
            )
            
            # Limit to 5 coordinates for quick testing
            if len(results) > 5:
                results = results.head(5)
            
            suitable_count = (results['prediction'] == 1).sum()
            total_count = len(results)
            print(f"  ✅ Success: {suitable_count}/{total_count} points predicted as suitable")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")

if __name__ == "__main__":
    # Test with existing models (10 coordinates only)
    test_inference_with_existing_models()
    
    # Comment out the different models test for now
    # test_with_different_models() 