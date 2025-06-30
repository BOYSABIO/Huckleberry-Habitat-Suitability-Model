"""
Inference pipeline for the Huckleberry Habitat Prediction Pipeline.
"""

import pandas as pd
import folium
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.config.settings import Settings
from src.utils.logging_config import get_logger, log_pipeline_step
from src.features.environmental import EnvironmentalDataExtractor
from src.models.registry import ModelRegistry
from src.data_validation.validate import validate_inference_data


class InferencePipeline:
    """Inference pipeline for making predictions."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize inference pipeline.
        
        Args:
            settings: Pipeline settings
        """
        self.settings = settings or Settings()
        self.logger = get_logger("inference_pipeline")
        self.model_registry = ModelRegistry(self.settings.model.model_registry_path)
        self.env_extractor = EnvironmentalDataExtractor()
        
        # Load current model
        self.model_data = None
        self.model = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """Load the current model from registry or specific file."""
        try:
            # Check if a specific model file is specified
            if self.settings.inference.model_file_path:
                import joblib
                model_file = Path(self.settings.inference.model_file_path)
                if not model_file.exists():
                    raise FileNotFoundError(f"Model file not found: {model_file}")
                
                loaded_data = joblib.load(model_file)
                
                # Handle different model file formats
                if isinstance(loaded_data, dict):
                    # Registry format: {'model': model, 'feature_names': [...], ...}
                    self.model_data = loaded_data
                    self.model = self.model_data['model']
                    self.feature_names = self.model_data['feature_names']
                    self.logger.info(f"Loaded registry-format model: {model_file}")
                elif isinstance(loaded_data, list) and len(loaded_data) > 0:
                    # Alternative format: [model, feature_names, ...]
                    self.model = loaded_data[0]
                    self.feature_names = loaded_data[1] if len(loaded_data) > 1 else []
                    self.model_data = {
                        'model': self.model,
                        'feature_names': self.feature_names,
                        'version_id': 'legacy_model'
                    }
                    self.logger.info(f"Loaded legacy-format model: {model_file}")
                    
                    # If feature_names is empty, try to get them from the model
                    if not self.feature_names and hasattr(self.model, 'feature_names_in_'):
                        self.feature_names = list(self.model.feature_names_in_)
                        self.model_data['feature_names'] = self.feature_names
                        self.logger.info(f"Extracted feature names from model: {self.feature_names}")
                else:
                    # Direct model format
                    self.model = loaded_data
                    self.feature_names = []
                    self.model_data = {
                        'model': self.model,
                        'feature_names': self.feature_names,
                        'version_id': 'direct_model'
                    }
                    self.logger.info(f"Loaded direct model: {model_file}")
                    
                    # Try to get feature names from the model
                    if hasattr(self.model, 'feature_names_in_'):
                        self.feature_names = list(self.model.feature_names_in_)
                        self.model_data['feature_names'] = self.feature_names
                        self.logger.info(f"Extracted feature names from model: {self.feature_names}")
            else:
                # Load from registry
                self.model_data = self.model_registry.load_model()
                self.model = self.model_data['model']
                self.feature_names = self.model_data['feature_names']
                self.logger.info(f"Loaded model: {self.model_data['version_id']}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _get_latest_gridmet_date(self) -> pd.Timestamp:
        """Get the latest available date from GridMET dataset."""
        try:
            # Load GridMET dataset to get time range
            ds = self.env_extractor.load_gridmet_dataset()
            latest_time = pd.to_datetime(ds.time.values.max())
            self.logger.info(f"Latest GridMET date: {latest_time}")
            return latest_time
        except Exception as e:
            self.logger.warning(f"Failed to get latest GridMET date: {e}")
            # Fallback to a reasonable date within GridMET range
            return pd.Timestamp('2020-12-31')
    
    @log_pipeline_step("Input Validation")
    def validate_input(
        self, 
        coordinates: List[Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Validate and prepare input data for inference.
        
        Args:
            coordinates: List of (lat, lon) tuples
            
        Returns:
            DataFrame with validated input data
        """
        self.logger.info(f"Validating input: {len(coordinates)} coordinates")
        
        # Create DataFrame from coordinates only
        # Let environmental extraction determine the best dates dynamically
        df = pd.DataFrame(coordinates, columns=['decimalLatitude', 'decimalLongitude'])
        
        self.logger.info("Coordinates prepared for environmental extraction")
        
        # Validate required columns
        is_valid = validate_inference_data(df, self.settings.inference.required_columns)
        if not is_valid:
            raise ValueError("Input validation failed")
        
        self.logger.info(f"Input validation passed: {len(df)} records")
        return df
    
    @log_pipeline_step("Environmental Data Extraction")
    def extract_environmental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract environmental data for inference coordinates."""
        self.logger.info("Extracting environmental data")
        
        # Extract GridMET data
        df_with_gridmet = self.env_extractor.extract_gridmet_data(df)
        self.logger.info(f"Records with GridMET data: {len(df_with_gridmet)}/{len(df)}")
        
        if len(df_with_gridmet) == 0:
            raise ValueError("No coordinates within GridMET bounds")
        
        # Add elevation data
        df_with_elevation = self.env_extractor.add_elevation_data(df_with_gridmet)
        
        # Add soil data
        df_with_soil = self.env_extractor.add_soil_data(df_with_elevation)
        
        # Extract year, month, day from datetime (added by environmental extraction)
        if 'datetime' in df_with_soil.columns:
            df_with_soil['year'] = df_with_soil['datetime'].dt.year
            df_with_soil['month'] = df_with_soil['datetime'].dt.month
            df_with_soil['day'] = df_with_soil['datetime'].dt.day
            
            # Add season_num feature (same logic as training pipeline)
            df_with_soil['season_num'] = df_with_soil['month'].apply(
                lambda m: 1 if m in [12, 1, 2] else  # Winter
                2 if m in [3, 4, 5] else  # Spring
                3 if m in [6, 7, 8] else  # Summer
                4  # Fall (9, 10, 11)
            )
            self.logger.info(f"Date components and season extracted from GridMET datetime")
        
        self.logger.info(f"Environmental data extraction complete: {df_with_soil.shape}")
        return df_with_soil
    
    @log_pipeline_step("Feature Preparation")
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction."""
        self.logger.info("Preparing features for prediction")
        
        # Debug: Log available columns and expected features
        self.logger.info(f"Available columns: {list(df.columns)}")
        self.logger.info(f"Model expects features: {self.feature_names}")
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            self.logger.error(f"Missing features: {missing_features}")
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select only the features used by the model
        feature_df = df[self.feature_names].copy()
        
        # Handle any missing values
        feature_df = feature_df.fillna(0)  # Simple imputation for inference
        
        self.logger.info(f"Feature preparation complete: {feature_df.shape}")
        return feature_df
    
    @log_pipeline_step("Model Prediction")
    def make_predictions(self, feature_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the loaded model."""
        self.logger.info("Making predictions")
        
        # Make predictions
        predictions = self.model.predict(feature_df)
        probabilities = self.model.predict_proba(feature_df)
        
        # Create results DataFrame with original data and predictions
        results_df = original_df.copy()
        results_df['prediction'] = predictions
        results_df['probability'] = probabilities[:, 1]  # Probability of occurrence
        
        self.logger.info(f"Predictions complete: {len(results_df)} records")
        return results_df
    
    def create_prediction_map(
        self,
        results_df: pd.DataFrame,
        output_path: str = "outputs/maps/prediction_map.html",
        confidence_threshold: float = 0.8
    ) -> str:
        """
        Create an interactive map showing predictions.
        
        Args:
            results_df: DataFrame with predictions
            output_path: Path to save the map
            confidence_threshold: Minimum confidence for suitable habitat
            
        Returns:
            Path to the created map file
        """
        self.logger.info("Creating prediction map")
        
        # Filter for suitable habitat
        suitable_df = results_df[results_df['probability'] >= confidence_threshold].copy()
        
        if len(suitable_df) == 0:
            self.logger.warning("No suitable habitat found above confidence threshold")
            return None
        
        # Create map
        map_center = [
            suitable_df['decimalLatitude'].mean(),
            suitable_df['decimalLongitude'].mean()
        ]
        
        m = folium.Map(location=map_center, zoom_start=8)
        
        # Add markers for suitable habitat
        for _, row in suitable_df.iterrows():
            popup_text = f"""
            <b>Suitable Huckleberry Habitat</b><br>
            Confidence: {row['probability']:.2%}<br>
            Latitude: {row['decimalLatitude']:.4f}<br>
            Longitude: {row['decimalLongitude']:.4f}<br>
            """
            
            folium.Marker(
                location=[row['decimalLatitude'], row['decimalLongitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='green', icon='leaf')
            ).add_to(m)
        
        # Save map
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        
        self.logger.info(f"Prediction map saved to: {output_path}")
        return str(output_path)
    
    def save_predictions_csv(
        self,
        results_df: pd.DataFrame,
        output_path: str = "outputs/predictions/inference_predictions.csv"
    ) -> str:
        """
        Save predictions to CSV file.
        
        Args:
            results_df: DataFrame with predictions
            output_path: Path to save the CSV file
            
        Returns:
            Path to the saved CSV file
        """
        self.logger.info("Saving predictions to CSV")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Predictions saved to: {output_path}")
        return str(output_path)
    
    def run(
        self,
        coordinates: List[Tuple[float, float]],
        create_map: bool = True,
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Run the complete inference pipeline.
        
        Args:
            coordinates: List of (lat, lon) tuples
            create_map: Whether to create a prediction map
            confidence_threshold: Minimum confidence for suitable habitat
            
        Returns:
            Dictionary with inference results
        """
        self.logger.info("Starting inference pipeline")
        
        try:
            # Step 1: Validate input
            df = self.validate_input(coordinates)
            
            # Step 2: Extract environmental data
            df = self.extract_environmental_data(df)
            
            # Step 3: Prepare features
            feature_df = self.prepare_features(df)
            
            # Step 4: Make predictions
            results_df = self.make_predictions(feature_df, df)
            
            # Step 5: Save predictions to CSV
            csv_path = self.save_predictions_csv(results_df)
            
            # Step 6: Create map (optional)
            map_path = None
            if create_map:
                map_path = self.create_prediction_map(
                    results_df, confidence_threshold=confidence_threshold
                )
            
            # Step 7: Prepare results
            suitable_count = sum(results_df['probability'] >= confidence_threshold)
            avg_confidence = results_df['probability'].mean()
            
            self.logger.info("✅ Inference pipeline completed successfully")
            
            return {
                "success": True,
                "total_coordinates": len(coordinates),
                "valid_coordinates": len(results_df),
                "suitable_habitat_count": suitable_count,
                "average_confidence": avg_confidence,
                "predictions": results_df,
                "csv_path": csv_path,
                "map_path": map_path,
                "model_version": self.model_data['version_id']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Inference pipeline failed: {str(e)}")
            raise 