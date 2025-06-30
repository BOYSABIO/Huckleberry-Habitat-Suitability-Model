"""
Simple test script for MLOps pipeline components.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_configuration():
    """Test configuration system."""
    print("🧪 Testing Configuration System...")
    
    try:
        from config.settings import Settings
        from config.environments import get_settings
        
        # Test default settings
        settings = Settings()
        assert settings.data.raw_data_path == "data/raw/occurrence.txt"
        assert settings.model.model_type == "random_forest"
        print("✅ Default settings loaded correctly")
        
        # Test environment-specific settings
        dev_settings = get_settings('development')
        assert dev_settings.model.model_name == "huckleberry_model_dev"
        print("✅ Development settings loaded correctly")
        
        prod_settings = get_settings('production')
        assert prod_settings.model.model_type == "ensemble"
        print("✅ Production settings loaded correctly")
        
        test_settings = get_settings('testing')
        assert test_settings.model.n_estimators == 10
        print("✅ Testing settings loaded correctly")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        raise


def test_logging():
    """Test logging configuration."""
    print("\n🧪 Testing Logging System...")
    
    try:
        from utils.logging_config import setup_logging
        
        logger = setup_logging(
            name="test_logger",
            level="DEBUG",
            log_file="logs/test.log"
        )
        
        logger.info("Test log message")
        logger.debug("Debug message")
        logger.warning("Warning message")
        
        # Check if log file was created
        log_file = Path("logs/test.log")
        assert log_file.exists()
        print("✅ Logging system working correctly")
        
    except Exception as e:
        print(f"❌ Logging test failed: {str(e)}")
        raise


def test_data_versioning():
    """Test data versioning system."""
    print("\n🧪 Testing Data Versioning...")
    
    try:
        from utils.data_versioning import DataVersioning
        
        # Create test data
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        versioning = DataVersioning("data/test_versions.json")
        
        # Track a transformation
        version_id = versioning.track_transformation(
            df=test_df,
            description="Test transformation",
            input_files=["test_input.csv"],
            output_files=["test_output.csv"],
            parameters={"param1": "value1"},
            metadata={"test": True}
        )
        
        assert version_id is not None
        print(f"✅ Data version created: {version_id}")
        
        # Get current version
        current_version = versioning.get_current_version()
        assert current_version is not None
        assert current_version["id"] == version_id
        print("✅ Current version retrieved correctly")
        
    except Exception as e:
        print(f"❌ Data versioning test failed: {str(e)}")
        raise


def test_model_registry():
    """Test model registry system."""
    print("\n🧪 Testing Model Registry...")
    
    try:
        from models.registry import ModelRegistry
        
        # Create test model (mock)
        class MockModel:
            def predict(self, X):
                return [1, 0, 1]
            
            def predict_proba(self, X):
                return [[0.2, 0.8], [0.9, 0.1], [0.3, 0.7]]
        
        mock_model = MockModel()
        
        registry = ModelRegistry("models/test_registry/")
        
        # Register a model
        version_id = registry.register_model(
            model=mock_model,
            model_name="test_model",
            model_type="random_forest",
            metrics={"accuracy": 0.85, "precision": 0.82},
            feature_names=["feature1", "feature2"],
            training_data_info={"total_records": 1000},
            parameters={"n_estimators": 100},
            description="Test model",
            tags=["test", "mock"]
        )
        
        assert version_id is not None
        print(f"✅ Model registered: {version_id}")
        
        # Load the model
        loaded_model_data = registry.load_model(version_id)
        assert loaded_model_data is not None
        assert loaded_model_data["version_id"] == version_id
        print("✅ Model loaded correctly")
        
        # Get current model
        current_model = registry.get_current_model()
        assert current_model is not None
        print("✅ Current model retrieved correctly")
        
    except Exception as e:
        print(f"❌ Model registry test failed: {str(e)}")
        raise


def test_pipeline_structure():
    """Test pipeline structure (without running full pipeline)."""
    print("\n🧪 Testing Pipeline Structure...")
    
    try:
        from config.environments import get_settings
        from pipelines.training_pipeline import TrainingPipeline
        
        # Use testing settings for faster execution
        settings = get_settings('testing')
        
        # Initialize pipeline
        pipeline = TrainingPipeline(settings)
        
        # Check that all components are initialized
        assert pipeline.data_loader is not None
        assert pipeline.preprocessor is not None
        assert pipeline.geocoder is not None
        assert pipeline.env_extractor is not None
        assert pipeline.model_registry is not None
        assert pipeline.data_versioning is not None
        
        print("✅ Training pipeline structure correct")
        
    except Exception as e:
        print(f"❌ Pipeline structure test failed: {str(e)}")
        raise


def main():
    """Run all tests."""
    print("🚀 Starting MLOps Pipeline Tests\n")
    
    try:
        # Test configuration
        test_configuration()
        
        # Test logging
        test_logging()
        
        # Test data versioning
        test_data_versioning()
        
        # Test model registry
        test_model_registry()
        
        # Test pipeline structure
        test_pipeline_structure()
        
        print("\n🎉 All MLOps pipeline tests completed!")
        print("\n📋 Summary:")
        print("- Configuration system: ✅ Working")
        print("- Logging system: ✅ Working")
        print("- Data versioning: ✅ Working")
        print("- Model registry: ✅ Working")
        print("- Pipeline structure: ✅ Working")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 