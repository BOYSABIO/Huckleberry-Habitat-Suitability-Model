#!/usr/bin/env python3
"""
Example script demonstrating habitat suitability prediction for a geographic area.
"""

import pandas as pd
from datetime import datetime
from src.inference.predict import predict_habitat_suitability

def main():
    """Example of predicting habitat suitability for Idaho huckleberry areas."""
    
    print("🌿 Huckleberry Habitat Suitability Prediction Example")
    print("=" * 60)
    
    # Example: Predict habitat suitability for a region in Idaho
    # This is a smaller area for demonstration (you can expand this)
    lat_min, lat_max = 44.5, 45.0  # Northern Idaho
    lon_min, lon_max = -116.5, -116.0
    
    print(f"Target area: ({lat_min:.3f}, {lon_min:.3f}) to ({lat_max:.3f}, {lon_max:.3f})")
    print(f"Resolution: 0.01 degrees (approximately 1km)")
    
    # Set target date (summer when huckleberries are typically found)
    target_date = datetime(2020, 7, 15)
    print(f"Target date: {target_date.strftime('%Y-%m-%d')}")
    
    # Model path (you'll need to train a model first)
    model_path = "models/huckleberry_model.joblib"
    
    try:
        # Run prediction
        print("\n🚀 Starting prediction...")
        results = predict_habitat_suitability(
            model_path=model_path,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            resolution=0.01,  # 1km resolution
            target_date=target_date,
            output_path="outputs/habitat_prediction_example.csv"
        )
        
        # Display results summary
        print("\n📊 Prediction Results:")
        print(f"Total grid points: {len(results)}")
        print(f"Predicted suitable habitat: {(results['prediction'] == 1).sum()}")
        print(f"Predicted unsuitable habitat: {(results['prediction'] == 0).sum()}")
        
        # Show some example predictions
        print("\n📍 Example Predictions:")
        sample_results = results.head(10)
        for _, row in sample_results.iterrows():
            status = "SUITABLE" if row['prediction'] == 1 else "UNSUITABLE"
            prob = row['probability_presence'] if row['prediction'] == 1 else row['probability_absence']
            print(f"  ({row['decimalLatitude']:.3f}, {row['decimalLongitude']:.3f}): "
                  f"{status} (confidence: {prob:.2f})")
        
        print(f"\n💾 Results saved to: outputs/habitat_prediction_example.csv")
        
    except FileNotFoundError:
        print(f"\n❌ Model file not found: {model_path}")
        print("Please train a model first using the main pipeline.")
        print("Example: python src/main.py --input data/raw/occurrence.txt --model ensemble")
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        print("Make sure you have:")
        print("1. A trained model file")
        print("2. Internet connection for environmental data extraction")
        print("3. Required dependencies installed")

if __name__ == "__main__":
    main() 