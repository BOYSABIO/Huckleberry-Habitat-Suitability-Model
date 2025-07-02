"""
Training pipeline for the Huckleberry Habitat Prediction Pipeline.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from src.config.settings import Settings
from src.utils.logging_config import get_logger, log_pipeline_step
from src.utils.data_versioning import DataVersioning
from src.data_load.loader import DataLoader
from src.data_preprocess.preprocessor import DataPreprocessor
from src.data_preprocess.geocode import Geocoder, load_manual_geocodes, apply_manual_geocodes
from src.data_validation.validate import validate_data
from src.features.environmental import EnvironmentalDataExtractor
from src.models.pipeline import HuckleberryPredictor, RandomForestPredictor
from src.models.registry import ModelRegistry


class TrainingPipeline:
    """Training pipeline orchestrator."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize training pipeline.
        
        Args:
            settings: Pipeline settings
        """
        self.settings = settings or Settings()
        self.logger = get_logger("training_pipeline")
        self.data_versioning = DataVersioning()
        self.model_registry = ModelRegistry(self.settings.model.model_registry_path)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.geocoder = Geocoder()
        self.env_extractor = EnvironmentalDataExtractor()
    
    @log_pipeline_step("Data Loading")
    def load_data(self) -> pd.DataFrame:
        """Load raw occurrence data."""
        self.logger.info(f"Loading data from: {self.settings.data.raw_data_path}")
        df = self.data_loader.load_gbif_occurrences(self.settings.data.raw_data_path)
        self.logger.info(f"Loaded {len(df)} raw records")
        return df
    
    @log_pipeline_step("Data Preprocessing")
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the raw data."""
        self.logger.info("Starting data preprocessing")
        
        # Basic cleaning
        df = self.preprocessor.clean_occurrence_data(df)
        self.logger.info(f"After cleaning: {len(df)} records")
        
        # Temporal filtering
        df = self.preprocessor.filter_gridmet_time_bounds(df)
        self.logger.info(f"After temporal filtering: {len(df)} records")
        
        return df
    
    @log_pipeline_step("Geocoding")
    def geocode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Geocode the occurrence data."""
        self.logger.info("Starting geocoding")
        
        # Geocode dataset
        df = self.geocoder.geocode_dataset(df)
        self.logger.info(f"After geocoding: {len(df)} records")
        
        # Spatial filtering
        df = self.preprocessor.filter_gridmet_bounds(df)
        self.logger.info(f"After spatial filtering: {len(df)} records")
        
        # Manual geocode fallback
        manual_dict = load_manual_geocodes()
        df = apply_manual_geocodes(df, manual_dict)
        self.logger.info(f"After manual geocode fallback: {len(df)} records")
        
        return df
    
    @log_pipeline_step("Environmental Data Extraction")
    def extract_environmental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract environmental data."""
        self.logger.info("Starting environmental data extraction")
        
        # Add occurrence column for real occurrences
        df['occurrence'] = 1
        
        # Extract GridMET data
        df_with_gridmet = self.env_extractor.extract_gridmet_data(df)
        self.logger.info(f"Records with GridMET data: {len(df_with_gridmet)}/{len(df)}")
        
        if len(df_with_gridmet) == 0:
            raise ValueError("No records within GridMET bounds. Cannot proceed.")
        
        return df_with_gridmet
    
    @log_pipeline_step("Pseudo-absence Generation")
    def generate_pseudo_absences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate pseudo-absences and combine with real occurrences."""
        self.logger.info("Generating pseudo-absences")
        
        combined_df = self.preprocessor.create_pseudo_absences(
            df,
            ratio=self.settings.data.pseudo_absence_ratio
        )
        
        self.logger.info(f"Generated {sum(combined_df['occurrence'] == 0)} pseudo-absences")
        self.logger.info(f"Total records: {len(combined_df)}")
        
        return combined_df
    
    @log_pipeline_step("Feature Engineering")
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional environmental features."""
        self.logger.info("Adding elevation and soil data")
        
        # Add elevation data
        df = self.env_extractor.add_elevation_data(df)
        
        # Add soil data
        df = self.env_extractor.add_soil_data(df)
        
        self.logger.info(f"Feature engineering complete: {df.shape}")
        return df
    
    @log_pipeline_step("Data Validation")
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the final dataset."""
        self.logger.info("Validating final dataset")
        
        expected_columns = [
            'decimalLatitude', 'decimalLongitude', 'year', 'month', 'day',
            'datetime', 'occurrence'
        ]
        
        is_valid = validate_data(df, expected_columns)
        if not is_valid:
            raise ValueError("Data validation failed")
        
        self.logger.info("Data validation passed")
        return True
    
    @log_pipeline_step("Model Training")
    def train_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the model."""
        self.logger.info(f"Training {self.settings.model.model_type} model")
        
        # Initialize model
        if self.settings.model.model_type == 'ensemble':
            model = HuckleberryPredictor()
        else:
            model = RandomForestPredictor(
                n_estimators=self.settings.model.n_estimators,
                random_state=self.settings.model.random_state
            )
        
        # Train model
        metrics = model.fit(
            df,
            target_col=self.settings.model.target_column,
            test_size=self.settings.model.test_size,
            random_state=self.settings.model.random_state
        )
        
        self.logger.info(f"Training completed. Metrics: {metrics}")
        return model, metrics
    
    @log_pipeline_step("Model Registration")
    def register_model(self, model, metrics: Dict[str, float], df: pd.DataFrame) -> str:
        """Register the trained model."""
        self.logger.info("Registering trained model")
        
        # Prepare training data info
        training_data_info = {
            "total_records": len(df),
            "occurrence_distribution": df[self.settings.model.target_column].value_counts().to_dict(),
            "feature_count": len(model.feature_names) if hasattr(model, 'feature_names') else None
        }
        
        # Register model
        version_id = self.model_registry.register_model(
            model=model,
            model_name=self.settings.model.model_name,
            model_type=self.settings.model.model_type,
            metrics=metrics,
            feature_names=model.feature_names if hasattr(model, 'feature_names') else [],
            training_data_info=training_data_info,
            parameters={
                "n_estimators": self.settings.model.n_estimators,
                "test_size": self.settings.model.test_size,
                "random_state": self.settings.model.random_state
            },
            description=f"Trained {self.settings.model.model_type} model",
            tags=["huckleberry", "habitat_prediction", self.settings.model.model_type]
        )
        
        self.logger.info(f"Model registered with version ID: {version_id}")
        return version_id
    
    @log_pipeline_step("Column Selection")
    def select_modeling_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only essential and environmental columns for modeling."""
        self.logger.info("Selecting modeling columns")
        
        # Essential columns that must be present (including target variable)
        essential_cols = [
            'decimalLatitude', 'decimalLongitude', 'year', 'month', 'day', 
            'datetime', 'occurrence', 'season_num'  # occurrence is the target variable
        ]
        
        # Environmental columns (GridMET, elevation, soil)
        environmental_cols = [
            'elevation', 'soil_ph', 'gridmet_lat', 'gridmet_lon', 'gridmet_date'
        ]
        
        # GridMET climate variables (actual variable names from EnvironmentalDataExtractor)
        gridmet_cols = [col for col in df.columns if col in [
            'air_temperature', 'precipitation_amount', 'specific_humidity', 'relative_humidity',
            'mean_vapor_pressure_deficit', 'potential_evapotranspiration', 
            'surface_downwelling_shortwave_flux_in_air', 'wind_speed'
        ]]
        
        # Combine all needed columns
        needed_cols = essential_cols + environmental_cols + gridmet_cols
        
        # Filter to only columns that exist in the dataframe
        available_cols = [col for col in needed_cols if col in df.columns]
        
        # Ensure occurrence column is present (critical for modeling)
        if 'occurrence' not in available_cols:
            self.logger.error("CRITICAL: occurrence column missing from available columns!")
            if 'occurrence' in df.columns:
                available_cols.append('occurrence')
                self.logger.info("Added occurrence column back to available columns")
            else:
                self.logger.error("occurrence column not found in dataframe!")
        
        # Add any missing essential columns with warnings
        missing_essential = [col for col in essential_cols if col not in available_cols]
        if missing_essential:
            self.logger.warning(f"Missing essential columns: {missing_essential}")
        
        # Select only the needed columns
        df_selected = df[available_cols].copy()
        
        # Verify occurrence column is present
        if 'occurrence' in df_selected.columns:
            occurrence_counts = df_selected['occurrence'].value_counts()
            self.logger.info(f"occurrence column present with distribution: {dict(occurrence_counts)}")
        else:
            self.logger.error("CRITICAL: occurrence column still missing after selection!")
        
        self.logger.info(f"Column selection: {len(df.columns)} -> {len(df_selected.columns)} columns")
        self.logger.info(f"Selected columns: {available_cols}")
        
        return df_selected
    
    @log_pipeline_step("Essential Column Selection")
    def select_essential_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only essential columns for processed data (before environmental extraction)."""
        self.logger.info("Selecting essential columns for processed data")
        
        # Essential columns only (no environmental data yet)
        essential_cols = [
            'gbifID', 'decimalLatitude', 'decimalLongitude', 'year', 'month', 'day', 
            'datetime', 'occurrence', 'season_num'
        ]
        
        # Filter to only columns that exist in the dataframe
        available_cols = [col for col in essential_cols if col in df.columns]
        
        # Add any missing essential columns with warnings
        missing_essential = [col for col in essential_cols if col not in df.columns]
        if missing_essential:
            self.logger.warning(f"Missing essential columns: {missing_essential}")
        
        # Select only the essential columns
        df_selected = df[available_cols].copy()
        
        self.logger.info(f"Essential column selection: {len(df.columns)} -> {len(df_selected.columns)} columns")
        self.logger.info(f"Selected essential columns: {available_cols}")
        
        return df_selected
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting training pipeline")
        
        try:
            # Step 1: Load data
            df = self.load_data()
            
            # Step 2: Preprocess data
            df = self.preprocess_data(df)
            
            # Step 3: Geocode data
            df = self.geocode_data(df)
            
            # Step 4: Generate pseudo-absences (before environmental extraction)
            df = self.generate_pseudo_absences(df)
            
            # Step 5: Extract environmental data (for all records including pseudo-absences)
            df = self.extract_environmental_data(df)
            
            # Step 6: Engineer features (elevation, soil)
            df = self.engineer_features(df)
            
            # Step 7: Validate data
            self.validate_data(df)
            
            # Step 8: Save processed data (essential columns only, before environmental extraction)
            df_essential = self.select_essential_columns(df)
            self.data_loader.save_processed_data(df_essential, 'huckleberry_processed.csv')
            self.logger.info(f"Processed data saved to: data/processed/huckleberry_processed.csv")
            
            # Step 9: Select modeling columns (essential + environmental)
            df = self.select_modeling_columns(df)
            
            # Step 10: Save enriched data (complete dataset with all environmental features)
            self.data_loader.save_enriched_data(df, 'huckleberry_final_enriched.csv')
            self.logger.info(f"Enriched data saved to: data/enriched/huckleberry_final_enriched.csv")
            
            # Step 11: Train model
            model, metrics = self.train_model(df)
            
            # Step 12: Register model
            version_id = self.register_model(model, metrics, df)
            
            # Step 13: Track data versioning
            version_id_data = self.data_versioning.track_transformation(
                df=df,
                description="Complete training pipeline run",
                input_files=[self.settings.data.raw_data_path],
                output_files=['huckleberry_processed.csv', 'huckleberry_final_enriched.csv'],
                parameters={
                    "pseudo_absence_ratio": self.settings.data.pseudo_absence_ratio,
                    "pseudo_absence_buffer_km": self.settings.data.pseudo_absence_buffer_km,
                    "model_type": self.settings.model.model_type,
                    "n_estimators": self.settings.model.n_estimators
                },
                metadata={
                    "model_version_id": version_id,
                    "training_metrics": metrics
                }
            )
            
            self.logger.info("Training pipeline completed successfully")
            
            return {
                "success": True,
                "model_version_id": version_id,
                "data_version_id": version_id_data,
                "metrics": metrics,
                "processed_data_path": 'huckleberry_processed.csv',
                "enriched_data_path": 'huckleberry_final_enriched.csv',
                "final_record_count": len(df)
            }
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise 