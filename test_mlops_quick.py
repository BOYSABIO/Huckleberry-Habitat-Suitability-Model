"""
Quick test to verify MLOps components are working.
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_basic_imports():
    """Test that basic imports work."""
    print("🧪 Testing Basic Imports...")
    
    try:
        # Test configuration
        from config.settings import Settings
        from config.environments import get_settings
        print("✅ Configuration imports work")
        
        # Test logging
        from utils.logging_config import setup_logging
        print("✅ Logging imports work")
        
        # Test data versioning
        from utils.data_versioning import DataVersioning
        print("✅ Data versioning imports work")
        
        # Test model registry
        from models.registry import ModelRegistry
        print("✅ Model registry imports work")
        
        # Test pipelines
        from pipelines.training_pipeline import TrainingPipeline
        from pipelines.inference_pipeline import InferencePipeline
        print("✅ Pipeline imports work")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {str(e)}")
        return False


def test_configuration():
    """Test configuration system."""
    print("\n🧪 Testing Configuration...")
    
    try:
        from config.settings import Settings
        from config.environments import get_settings
        
        # Test default settings
        settings = Settings()
        assert settings.data.raw_data_path == "data/raw/occurrence.txt"
        print("✅ Default settings work")
        
        # Test environment settings
        dev_settings = get_settings('development')
        assert dev_settings.model.model_name == "huckleberry_model_dev"
        print("✅ Environment settings work")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False


def test_logging():
    """Test logging system."""
    print("\n🧪 Testing Logging...")
    
    try:
        from utils.logging_config import setup_logging
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        logger = setup_logging(
            name="test_logger",
            level="INFO",
            log_file="logs/test.log"
        )
        
        logger.info("Test message")
        print("✅ Logging system works")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {str(e)}")
        return False


def main():
    """Run quick tests."""
    print("🚀 Quick MLOps Test\n")
    
    tests = [
        test_basic_imports,
        test_configuration,
        test_logging
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MLOps structure is working.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 