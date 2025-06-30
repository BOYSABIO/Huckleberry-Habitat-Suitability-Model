"""
Central configuration settings for the Huckleberry Habitat Prediction Pipeline.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DataSettings:
    """Data loading and processing settings."""
    # Input/Output paths
    raw_data_path: str = "data/raw/occurrence.txt"
    processed_data_path: str = "data/processed/"
    enriched_data_path: str = "data/enriched/"
    
    # Data processing parameters
    pseudo_absence_ratio: int = 3
    pseudo_absence_buffer_km: float = 5.0
    random_seed: int = 42
    
    # Temporal filtering
    min_year: int = 1979
    max_year: int = 2020
    
    # Spatial filtering (GridMET bounds)
    min_lat: float = 25.0
    max_lat: float = 52.0
    min_lon: float = -125.0
    max_lon: float = -65.0


@dataclass
class ModelSettings:
    """Model training and evaluation settings."""
    # Model parameters
    model_type: str = "random_forest"  # "random_forest" or "ensemble"
    n_estimators: int = 100
    test_size: float = 0.2
    random_state: int = 42
    
    # Model storage
    model_registry_path: str = "models/"
    model_name: str = "huckleberry_model"
    
    # Feature settings
    target_column: str = "occurrence"
    excluded_columns: list = None
    
    def __post_init__(self):
        if self.excluded_columns is None:
            self.excluded_columns = ['gbifID', 'decimalLatitude', 'decimalLongitude', 'datetime', 'gridmet_date']


@dataclass
class InferenceSettings:
    """Inference pipeline settings."""
    # Input requirements
    required_columns: list = None
    coordinate_columns: list = None
    
    # Model loading
    model_file_path: Optional[str] = None  # Specific model file to use
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = [
                'decimalLatitude', 'decimalLongitude', 'year', 'month', 'day'
            ]
        if self.coordinate_columns is None:
            self.coordinate_columns = ['decimalLatitude', 'decimalLongitude']


@dataclass
class LoggingSettings:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = "logs/pipeline.log"


@dataclass
class Settings:
    """Main settings class that combines all configuration."""
    data: DataSettings = None
    model: ModelSettings = None
    inference: InferenceSettings = None
    logging: LoggingSettings = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataSettings()
        if self.model is None:
            self.model = ModelSettings()
        if self.inference is None:
            self.inference = InferenceSettings()
        if self.logging is None:
            self.logging = LoggingSettings()
    
    def get_model_path(self, model_name: Optional[str] = None) -> str:
        """Get the full path for a model file."""
        if model_name is None:
            model_name = self.model.model_name
        return str(Path(self.model.model_registry_path) / f"{model_name}.joblib")
    
    def get_processed_data_path(self, filename: str) -> str:
        """Get the full path for processed data."""
        return str(Path(self.data.processed_data_path) / filename)
    
    def get_enriched_data_path(self, filename: str) -> str:
        """Get the full path for enriched data."""
        return str(Path(self.data.enriched_data_path) / filename)


# Default settings instance
default_settings = Settings() 