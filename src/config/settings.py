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
    
    def __post_init__(self):
        """Validate settings after initialization."""
        if self.pseudo_absence_ratio < 1:
            raise ValueError("pseudo_absence_ratio must be >= 1")
        if self.pseudo_absence_buffer_km < 0:
            raise ValueError("pseudo_absence_buffer_km must be >= 0")
        if self.min_year > self.max_year:
            raise ValueError("min_year cannot be greater than max_year")
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be less than max_lat")
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be less than max_lon")


@dataclass
class ModelSettings:
    """Model training and evaluation settings."""
    # Model parameters
    model_type: str = "random_forest"  # "random_forest" or "ensemble" (random_forest is default)
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
    
    # GridMET date settings
    use_latest_gridmet: bool = True  # Use latest available GridMET data
    gridmet_date: Optional[str] = None  # Specific date for GridMET data (YYYY-MM-DD format)
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = [
                'decimalLatitude', 'decimalLongitude'
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