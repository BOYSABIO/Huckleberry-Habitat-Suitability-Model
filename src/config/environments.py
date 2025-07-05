"""
Environment-specific configuration settings.
"""

import os
from typing import Optional
from .settings import Settings, DataSettings, ModelSettings, InferenceSettings, LoggingSettings


def get_settings(environment: Optional[str] = None) -> Settings:
    """
    Get settings for a specific environment.
    
    Args:
        environment: Environment name ('development', 'production', 'testing', 'test_sample')
        
    Returns:
        Settings object configured for the environment
    """
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development')
    
    if environment == 'production':
        return _get_production_settings()
    elif environment == 'testing':
        return _get_testing_settings()
    elif environment == 'test_sample':
        return _get_test_sample_settings()
    else:  # development
        return _get_development_settings()


def _get_development_settings() -> Settings:
    """Development environment settings."""
    return Settings(
        data=DataSettings(
            raw_data_path="data/raw/occurrence_test_sample.txt",  # Use test sample for fast development
            processed_data_path="data/processed/",
            enriched_data_path="data/enriched/",
            pseudo_absence_ratio=3,
            pseudo_absence_buffer_km=5.0,
            random_seed=42
        ),
        model=ModelSettings(
            model_type="random_forest",  # Default to random forest
            n_estimators=100,
            test_size=0.2,
            random_state=42,
            model_registry_path="models/",
            model_name="huckleberry_model_dev"
        ),
        inference=InferenceSettings(
            model_file_path="models/random_forest_improved.joblib"  # Use improved random forest
        ),
        logging=LoggingSettings(
            level="DEBUG",
            log_file="logs/pipeline_dev.log"
        )
    )


def _get_production_settings() -> Settings:
    """Production environment settings."""
    return Settings(
        data=DataSettings(
            raw_data_path="data/raw/occurrence.txt",
            processed_data_path="data/processed/",
            enriched_data_path="data/enriched/",
            pseudo_absence_ratio=3,
            pseudo_absence_buffer_km=5.0,
            random_seed=42
        ),
        model=ModelSettings(
            model_type="random_forest",  # Default to random forest for production
            n_estimators=200,
            test_size=0.2,
            random_state=42,
            model_registry_path="models/",
            model_name="huckleberry_model_prod"
        ),
        inference=InferenceSettings(
            model_file_path=None  # Will auto-select latest production model
        ),
        logging=LoggingSettings(
            level="INFO",
            log_file="logs/pipeline_prod.log"
        )
    )


def _get_testing_settings() -> Settings:
    """Testing environment settings."""
    return Settings(
        data=DataSettings(
            raw_data_path="data/test/test_occurrence.txt",
            processed_data_path="data/test/processed/",
            enriched_data_path="data/test/enriched/",
            pseudo_absence_ratio=2,
            pseudo_absence_buffer_km=2.0,
            random_seed=42
        ),
        model=ModelSettings(
            model_type="random_forest",
            n_estimators=10,  # Smaller for faster testing
            test_size=0.3,
            random_state=42,
            model_registry_path="models/test/",
            model_name="huckleberry_model_test"
        ),
        inference=InferenceSettings(),
        logging=LoggingSettings(
            level="DEBUG",
            log_file="logs/pipeline_test.log"
        )
    )


def _get_test_sample_settings() -> Settings:
    """Test sample environment settings (uses actual test sample data)."""
    return Settings(
        data=DataSettings(
            raw_data_path="data/raw/occurrence_test_sample.txt",
            processed_data_path="data/test_sample/processed/",
            enriched_data_path="data/test_sample/enriched/",
            pseudo_absence_ratio=2,
            pseudo_absence_buffer_km=2.0,
            random_seed=42
        ),
        model=ModelSettings(
            model_type="random_forest",
            n_estimators=50,  # Medium size for test sample
            test_size=0.2,
            random_state=42,
            model_registry_path="models/test_sample/",
            model_name="huckleberry_model_test_sample"
        ),
        inference=InferenceSettings(),
        logging=LoggingSettings(
            level="DEBUG",
            log_file="logs/pipeline_test_sample.log"
        )
    ) 