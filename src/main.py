"""
Main orchestrator for the Huckleberry Habitat Prediction Pipeline.

This script ties together all modular components: loading, preprocessing,
geocoding (with manual fallback), validation, feature engineering, model
training, and inference.
"""

import argparse
import logging
import pandas as pd
from pathlib import Path

from data_load.loader import DataLoader
from data_preprocess.preprocessor import DataPreprocessor
from data_preprocess.geocode import Geocoder, load_manual_geocodes, apply_manual_geocodes
from data_validation.validate import validate_data
from features.environmental import EnvironmentalDataExtractor
from model.pipeline import HuckleberryPredictor, RandomForestPredictor
from model.feature_importance import FeatureAnalyzer
from inference.predict import run_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(args):
    # 1. Load data
    loader = DataLoader()
    df = loader.load_gbif_occurrences(args.input)
    logger.info(f"Loaded raw data: {len(df)} records")

    # 2. Preprocess (basic cleaning and date processing)
    preprocessor = DataPreprocessor()
    df = preprocessor.clean_occurrence_data(df)
    logger.info(f"After cleaning: {len(df)} records")

    # 3. Temporal filtering (gridMET time bounds)
    df = preprocessor.filter_gridmet_time_bounds(df)
    logger.info(f"After temporal filtering: {len(df)} records")

    # 4. Geocode (only temporally valid records)
    geocoder = Geocoder()
    df = geocoder.geocode_dataset(df)
    logger.info(f"After geocoding: {len(df)} records")

    # 5. Spatial filtering (gridMET spatial bounds)
    df = preprocessor.filter_gridmet_bounds(df)
    logger.info(f"After spatial filtering: {len(df)} records")

    # 6. Manual geocode fallback
    manual_dict = load_manual_geocodes()
    df = apply_manual_geocodes(df, manual_dict)
    logger.info(f"After manual geocode fallback: {len(df)} records")

    # 7. Validate
    validate_data(df, expected_columns=['decimallatitude', 'decimallongitude'])

    # 8. Add occurrence column (for real occurrences)
    df['occurrence'] = 1
    logger.info("Added 'occurrence' column (set to 1 for all records)")

    # 9. Feature engineering
    env = EnvironmentalDataExtractor()
    df = env.add_elevation_data(df)
    df = env.add_soil_data(df)
    df = env.create_environmental_features(df)
    logger.info(f"After feature engineering: {df.shape}")

    # 10. Model training
    if args.model == 'ensemble':
        model = HuckleberryPredictor()
    else:
        model = RandomForestPredictor()
    X = df.drop(columns=['occurrence'])
    y = df['occurrence']
    model.fit(X, y)
    model.save_model(args.model_path)
    logger.info(f"Model trained and saved to {args.model_path}")

    # 11. Inference (optional)
    if args.infer:
        preds = run_inference(args.model_path, X)
        logger.info(f"Inference complete. Predictions: {preds[:5]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huckleberry Habitat Prediction Pipeline")
    parser.add_argument('--input', type=str, default='data/raw/occurrence.txt', help='Input data file')
    parser.add_argument('--model', type=str, choices=['ensemble', 'random_forest'], default='ensemble', help='Model type')
    parser.add_argument('--model-path', type=str, default='models/huckleberry_model.joblib', help='Path to save trained model')
    parser.add_argument('--infer', action='store_true', help='Run inference after training')
    args = parser.parse_args()
    run_pipeline(args) 