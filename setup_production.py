#!/usr/bin/env python3
"""
Setup script for production version of Huckleberry Habitat Prediction System.

This script helps organize the existing codebase and prepare it for production use.
"""

import os
import shutil
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create the production directory structure."""
    directories = [
        "models",
        "analysis", 
        "logs",
        "src/models",
        "src/data",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def organize_existing_scripts():
    """Organize existing scripts into the new structure."""
    
    # Move the best pipeline to models directory
    if os.path.exists("notebooks/best_pipeline.py"):
        shutil.copy("notebooks/best_pipeline.py", "src/models/best_pipeline_backup.py")
        logger.info("Backed up best_pipeline.py")
    
    # Create __init__.py files for Python packages
    init_files = [
        "src/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            Path(init_file).touch()
            logger.info(f"Created {init_file}")


def create_config_file():
    """Create a configuration file for the system."""
    config_content = """# Configuration file for Huckleberry Habitat Prediction System

[DATA]
# Paths to data files
raw_data_path = data/raw/
processed_data_path = data/processed/
enriched_data_path = data/enriched/

[MODELS]
# Model configuration
default_model_type = ensemble
test_size = 0.2
random_state = 42

[OUTPUT]
# Output directories
models_dir = models/
analysis_dir = analysis/
logs_dir = logs/

[LOGGING]
# Logging configuration
log_level = INFO
log_file = logs/huckleberry_prediction.log
"""

    with open("config.ini", "w") as f:
        f.write(config_content)
    
    logger.info("Created config.ini")


def create_test_script():
    """Create a basic test script."""
    test_content = '''#!/usr/bin/env python3
"""
Basic tests for the Huckleberry Habitat Prediction System.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Import our modules
try:
    from src.models.pipeline import HuckleberryPredictor
    from src.models.feature_importance import FeatureAnalyzer
except ImportError:
    print("Warning: Could not import modules. Make sure they are properly installed.")


class TestHuckleberryPredictor(unittest.TestCase):
    """Test cases for the main predictor."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        self.test_data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'feature_5': np.random.randn(n_samples),
            'occurrence': np.random.randint(0, 2, n_samples)
        })
    
    def test_predictor_initialization(self):
        """Test that predictor can be initialized."""
        try:
            predictor = HuckleberryPredictor()
            self.assertIsNotNone(predictor)
        except Exception as e:
            self.fail(f"Failed to initialize predictor: {e}")
    
    def test_data_preparation(self):
        """Test data preparation functionality."""
        try:
            predictor = HuckleberryPredictor()
            features, target = predictor.prepare_data(self.test_data)
            
            self.assertEqual(len(features.columns), 5)
            self.assertEqual(len(target), 100)
            self.assertIn('occurrence', self.test_data.columns)
        except Exception as e:
            self.fail(f"Failed to prepare data: {e}")


class TestFeatureAnalyzer(unittest.TestCase):
    """Test cases for feature analyzer."""
    
    def test_analyzer_initialization(self):
        """Test that analyzer can be initialized."""
        try:
            analyzer = FeatureAnalyzer()
            self.assertIsNotNone(analyzer)
        except Exception as e:
            self.fail(f"Failed to initialize analyzer: {e}")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
'''
    
    with open("tests/test_basic.py", "w") as f:
        f.write(test_content)
    
    logger.info("Created basic test script")


def create_gitignore():
    """Create a .gitignore file for the project."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
models/*.joblib
analysis/*.csv
analysis/*.json
logs/*.log
*.log

# Data (uncomment if you don't want to track data files)
# data/
# *.csv
# *.json
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    logger.info("Created .gitignore")


def main():
    """Main setup function."""
    logger.info("Setting up production version of Huckleberry Habitat Prediction System")
    
    # Create directory structure
    create_directory_structure()
    
    # Organize existing scripts
    organize_existing_scripts()
    
    # Create configuration file
    create_config_file()
    
    # Create test script
    create_test_script()
    
    # Create .gitignore
    create_gitignore()
    
    logger.info("\\nSetup complete!")
    logger.info("\\nNext steps:")
    logger.info("1. Install dependencies: pip install -r requirements.txt")
    logger.info("2. Run example: python example_usage.py")
    logger.info("3. Run tests: python -m pytest tests/")
    logger.info("4. Use CLI: python main.py --help")


if __name__ == "__main__":
    main() 