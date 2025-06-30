"""
Inference module for huckleberry pipeline.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Any, Tuple, List, Optional
from pathlib import Path
from datetime import datetime

# Import our pipeline components
from src.features.environmental import EnvironmentalDataExtractor
from src.data_preprocess.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

def create_prediction_grid(lat_min: float, lat_max: float, lon_min: float, lon_max: float, 
                          resolution: float = 0.01) -> pd.DataFrame:
    """
    Create a grid of coordinates for prediction.
    
    Args:
        lat_min, lat_max: Latitude bounds
        lon_min, lon_max: Longitude bounds  
        resolution: Grid resolution in degrees (default 0.01 ≈ 1km)
    
    Returns:
        DataFrame with lat/lon coordinates
    """
    logger.info(f"Creating prediction grid: {lat_min:.3f} to {lat_max:.3f} lat, "
               f"{lon_min:.3f} to {lon_max:.3f} lon, resolution {resolution}")
    
    # Create coordinate arrays
    lats = np.arange(lat_min, lat_max + resolution, resolution)
    lons = np.arange(lon_min, lon_max + resolution, resolution)
    
    # Create grid
    lat_grid, lon_grid = np.meshgrid(lats, lons)
    
    # Flatten and create DataFrame
    grid_df = pd.DataFrame({
        'decimalLatitude': lat_grid.flatten(),
        'decimalLongitude': lon_grid.flatten()
    })
    
    logger.info(f"Created grid with {len(grid_df)} points")
    return grid_df

def safe_datetime_parse(datetime_series):
    """Safely parse datetime series with mixed formats"""
    parsed_dates = []
    
    for date_str in datetime_series:
        try:
            # Try parsing as full datetime first
            parsed = pd.to_datetime(date_str)
            parsed_dates.append(parsed)
        except:
            try:
                # If that fails, try parsing as date only
                parsed = pd.to_datetime(date_str, format='%Y-%m-%d')
                parsed_dates.append(parsed)
            except:
                # If all else fails, use NaT (Not a Time)
                parsed_dates.append(pd.NaT)
    
    return pd.Series(parsed_dates, index=datetime_series.index)

def month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def prepare_inference_data(coordinates_df: pd.DataFrame, 
                          target_date: Optional[datetime] = None,
                          use_latest_gridmet: bool = True) -> pd.DataFrame:
    """
    Prepare inference data with the same structure as training data.
    
    Args:
        coordinates_df: DataFrame with decimalLatitude and decimalLongitude columns
        target_date: Date for environmental data extraction
        use_latest_gridmet: Whether to use latest available GridMET data
    
    Returns:
        DataFrame ready for inference with same columns as training data
    """
    logger.info(f"Preparing inference data for {len(coordinates_df)} coordinates")
    
    # Create a copy to avoid modifying the original
    inference_df = coordinates_df.copy()
    
    # Add required columns for environmental extraction
    if target_date is None:
        # Use latest available date (you might want to make this configurable)
        target_date = datetime(2020, 7, 15)  # Example date
    
    inference_df['datetime'] = target_date
    
    # Add dummy columns required by environmental extractor
    inference_df['countryCode'] = 'US'
    inference_df['stateProvince'] = 'Unknown'
    inference_df['county'] = 'Unknown'
    
    # Extract environmental data
    env_extractor = EnvironmentalDataExtractor()
    
    # Extract GridMET data
    logger.info("Extracting GridMET data...")
    inference_df = env_extractor.extract_gridmet_data(inference_df)
    
    if len(inference_df) == 0:
        raise ValueError("No coordinates within GridMET bounds")
    
    # Add elevation data
    logger.info("Adding elevation data...")
    inference_df = env_extractor.add_elevation_data(inference_df)
    
    # Add soil data
    logger.info("Adding soil data...")
    inference_df = env_extractor.add_soil_data(inference_df)
    
    # Debug: Check what columns we have after environmental extraction
    logger.info(f"Columns after environmental extraction: {list(inference_df.columns)}")
    
    # Parse datetime using the same method as training
    inference_df['parsed_datetime'] = safe_datetime_parse(inference_df['datetime'])
    
    # Extract datetime components (same as training)
    inference_df['year'] = inference_df['parsed_datetime'].dt.year
    inference_df['month'] = inference_df['parsed_datetime'].dt.month
    inference_df['day'] = inference_df['parsed_datetime'].dt.day
    
    # Create season feature (same as training)
    inference_df['season'] = inference_df['month'].apply(month_to_season)
    season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
    inference_df['season_num'] = inference_df['season'].map(season_map)
    inference_df.drop(columns=['season'], inplace=True)
    
    # Drop the same columns that were dropped during training
    # Based on the training code: ['gbifID', 'gridmet_lat', 'gridmet_lon', 'gridmet_date', 'decimalLatitude', 'decimalLongitude', 'datetime', 'parsed_datetime']
    columns_to_drop = ['gbifID', 'gridmet_lat', 'gridmet_lon', 'gridmet_date', 'decimalLatitude', 'decimalLongitude', 'datetime', 'parsed_datetime']
    inference_df = inference_df.drop(columns=[col for col in columns_to_drop if col in inference_df.columns])
    
    # Also drop other columns that weren't in training data
    additional_drops = ['countryCode', 'stateProvince', 'county']
    inference_df = inference_df.drop(columns=[col for col in additional_drops if col in inference_df.columns])
    
    # Ensure we have the exact same columns as training data in the same order
    # Based on the training data structure: year, month, day, air_temperature, precipitation_amount, 
    # specific_humidity, relative_humidity, mean_vapor_pressure_deficit, potential_evapotranspiration, 
    # surface_downwelling_shortwave_flux_in_air, wind_speed, elevation, soil_ph, season_num
    # Note: 'occurrence' was the target variable, not a feature
    expected_columns = [
        'year', 'month', 'day', 'air_temperature', 'precipitation_amount', 'specific_humidity',
        'relative_humidity', 'mean_vapor_pressure_deficit', 'potential_evapotranspiration',
        'surface_downwelling_shortwave_flux_in_air', 'wind_speed', 
        'elevation', 'soil_ph', 'season_num'
    ]
    
    # Add missing columns with default values
    for col in expected_columns:
        if col not in inference_df.columns:
            inference_df[col] = 0.0  # Default numeric value
    
    # Reorder columns to match training data exactly
    inference_df = inference_df[expected_columns]
    
    logger.info(f"Inference data prepared: {inference_df.shape}")
    logger.info(f"Features: {list(inference_df.columns)}")
    
    return inference_df

def predict_habitat_suitability(model_path: str, 
                              lat_min: float, lat_max: float, 
                              lon_min: float, lon_max: float,
                              resolution: float = 0.01,
                              target_date: Optional[datetime] = None,
                              output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Predict habitat suitability for a geographic area.
    
    Args:
        model_path: Path to trained model
        lat_min, lat_max, lon_min, lon_max: Geographic bounds
        resolution: Grid resolution in degrees
        target_date: Date for environmental data (if None, uses latest)
        output_path: Path to save results (optional)
    
    Returns:
        DataFrame with coordinates and predictions
    """
    logger.info(f"Starting habitat suitability prediction for area: "
               f"({lat_min:.3f}, {lon_min:.3f}) to ({lat_max:.3f}, {lon_max:.3f})")
    
    # Create prediction grid
    grid_df = create_prediction_grid(lat_min, lat_max, lon_min, lon_max, resolution)
    
    # Store original coordinates for results
    original_coords = grid_df[['decimalLatitude', 'decimalLongitude']].copy()
    
    # Prepare inference data
    inference_df = prepare_inference_data(grid_df, target_date)
    
    # Load model and make predictions
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Get the exact feature names the model was trained with
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        logger.info(f"Model expects features: {list(expected_features)}")
    else:
        logger.warning("Model doesn't have feature_names_in_ attribute, using fallback")
        expected_features = None
    
    # Use model's prepare_inference_data method if available
    if hasattr(model, 'prepare_inference_data'):
        prediction_features = model.prepare_inference_data(inference_df)
        predictions = model.predict(prediction_features)
        probabilities = model.predict_proba(prediction_features)
    else:
        # Fallback for older models - ensure exact feature order
        if expected_features is not None:
            # Reorder inference_df to match model's expected features exactly
            missing_features = set(expected_features) - set(inference_df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    inference_df[feature] = 0.0  # Add missing features with default values
            
            # Reorder columns to match model's expected order exactly
            prediction_features = inference_df[expected_features]
        else:
            # Fallback for older models
            prediction_features = inference_df.drop(['gbifID', 'decimalLatitude', 
                                                   'decimalLongitude'], axis=1, errors='ignore')
        
        predictions = model.predict(prediction_features)
        probabilities = model.predict_proba(prediction_features)
    
    # Create results DataFrame with original coordinates
    results_df = original_coords.copy()
    results_df['prediction'] = predictions
    results_df['probability_absence'] = probabilities[:, 0]  # Probability of absence
    results_df['probability_presence'] = probabilities[:, 1]  # Probability of presence
    
    # Add environmental features to results (if they exist in inference_df)
    env_cols = ['elevation', 'soil_ph', 'air_temperature', 'precipitation_amount', 
                'season_num']
    for col in env_cols:
        if col in inference_df.columns:
            results_df[col] = inference_df[col]
    
    # Save results if requested
    if output_path:
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    
    logger.info(f"Prediction complete: {len(results_df)} points processed")
    logger.info(f"Prediction distribution: {results_df['prediction'].value_counts().to_dict()}")
    
    return results_df

def create_prediction_map(results_df: pd.DataFrame, 
                         output_path: str = "outputs/prediction_map.html", 
                         confidence_threshold: float = 0.8) -> None:
    """
    Create an HTML map showing high-confidence suitable habitat predictions.
    
    Args:
        results_df: DataFrame with prediction results
        output_path: Path to save the HTML map
        confidence_threshold: Minimum confidence to show on map (default 0.8 = 80%)
    """
    try:
        import folium
        
        # Filter for high-confidence SUITABLE predictions only
        high_conf_df = results_df[
            (results_df['prediction'] == 1) & 
            (results_df['probability_presence'] >= confidence_threshold)
        ].copy()
        
        if len(high_conf_df) == 0:
            logger.warning(f"No suitable habitat predictions with {confidence_threshold*100}%+ confidence found")
            return
        
        logger.info(f"Creating map with {len(high_conf_df)} high-confidence suitable habitat predictions")
        
        # Calculate center of the area
        center_lat = high_conf_df['decimalLatitude'].mean()
        center_lon = high_conf_df['decimalLongitude'].mean()
        
        # Create the map
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=10, 
            tiles='cartodbpositron'
        )
        
        # Add markers for each high-confidence suitable prediction
        for _, row in high_conf_df.iterrows():
            lat, lon = row['decimalLatitude'], row['decimalLongitude']
            prob = row['probability_presence']
            
            # Create popup content
            popup_content = f"""
            <b>Huckleberry Habitat Prediction</b><br>
            Status: SUITABLE<br>
            Confidence: {prob:.1%}<br>
            Coordinates: ({lat:.4f}, {lon:.4f})<br>
            """
            
            # Add environmental info if available
            if 'elevation' in row:
                popup_content += f"Elevation: {row['elevation']:.0f}m<br>"
            if 'soil_ph' in row:
                popup_content += f"Soil pH: {row['soil_ph']:.1f}<br>"
            if 'air_temperature' in row:
                popup_content += f"Temperature: {row['air_temperature']:.1f}°C<br>"
            if 'precipitation_amount' in row:
                popup_content += f"Precipitation: {row['precipitation_amount']:.1f}mm<br>"
            if 'season_num' in row:
                seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
                popup_content += f"Season: {seasons.get(row['season_num'], 'Unknown')}<br>"
            
            # Add circle marker (green for suitable habitat)
            marker = folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                popup=popup_content,
                color='green',
                fill=True,
                fillColor='green',
                fillOpacity=0.7,
                weight=2
            )
            marker.add_to(m)
        
        # Save the map
        m.save(output_path)
        logger.info(f"Map saved to: {output_path}")
        logger.info(f"Showing {len(high_conf_df)} suitable habitat predictions with {confidence_threshold*100}%+ confidence")
        
        # Show summary
        total_predictions = len(results_df)
        suitable_predictions = (results_df['prediction'] == 1).sum()
        high_conf_suitable = len(high_conf_df)
        
        logger.info(f"Summary: {total_predictions} total predictions, {suitable_predictions} suitable, {high_conf_suitable} high-confidence suitable")
        
    except ImportError:
        logger.error("Folium not installed. Install with: pip install folium")
    except Exception as e:
        logger.error(f"Error creating map: {e}")

def run_inference(model_path: str, data: pd.DataFrame) -> Any:
    """
    Run inference using a trained model (legacy function for backward compatibility).
    Args:
        model_path: Path to the trained model file
        data: DataFrame with features for prediction
    Returns:
        Model predictions
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Running inference on {len(data)} samples")
    
    # Use the model's prepare_inference_data method if available
    if hasattr(model, 'prepare_inference_data'):
        inference_data = model.prepare_inference_data(data)
        predictions = model.predict(inference_data)
    else:
        # Fallback for older models
        predictions = model.predict(data)
    
    return predictions

def run_inference_with_probabilities(model_path: str, data: pd.DataFrame) -> Tuple[Any, Any]:
    """
    Run inference and get both predictions and probabilities.
    Args:
        model_path: Path to the trained model file
        data: DataFrame with features for prediction
    Returns:
        Tuple of (predictions, probabilities)
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Running inference on {len(data)} samples")
    
    # Use the model's prepare_inference_data method if available
    if hasattr(model, 'prepare_inference_data'):
        inference_data = model.prepare_inference_data(data)
        predictions = model.predict(inference_data)
        probabilities = model.predict_proba(inference_data)
    else:
        # Fallback for older models
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)
    
    return predictions, probabilities

def run_inference_on_file(model_path: str, data_path: str, 
                         output_path: str = None) -> pd.DataFrame:
    """
    Run inference on a CSV file and save results (legacy function).
    Args:
        model_path: Path to the trained model file
        data_path: Path to the input CSV file
        output_path: Path to save results (optional)
    Returns:
        DataFrame with original data and predictions
    """
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Run inference
    predictions, probabilities = run_inference_with_probabilities(model_path, data)
    
    # Add predictions to the data
    result_df = data.copy()
    result_df['prediction'] = predictions
    result_df['probability_0'] = probabilities[:, 0]  # Probability of absence
    result_df['probability_1'] = probabilities[:, 1]  # Probability of presence
    
    # Save results if output path provided
    if output_path:
        result_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    
    return result_df 