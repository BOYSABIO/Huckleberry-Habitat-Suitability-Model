# Huckleberry Habitat Prediction Pipeline

A comprehensive MLOps pipeline for predicting huckleberry habitat suitability using environmental data, machine learning, and geospatial analysis.

## 🎯 Project Overview

This project analyzes and predicts huckleberry habitat suitability by combining:
- **Historical occurrence data** from GBIF (Global Biodiversity Information Facility)
- **Environmental variables** from GridMET (climate data)
- **Elevation data** from Open-Elevation API
- **Soil pH data** from SoilGrids API
- **Machine learning models** (Random Forest and Ensemble)

The pipeline automatically extracts environmental data, generates pseudo-absences for balanced training, and provides both training and inference capabilities with organized outputs.

## 🏗️ Project Structure

```
Capstone-Microsoft/
├── data/
│   ├── raw/                    # Original GBIF occurrence data
│   ├── processed/              # Cleaned and processed data
│   ├── enriched/               # Data with environmental features
│   └── test/                   # Test datasets
├── models/                     # Trained models and registry
├── outputs/
│   ├── maps/                   # HTML prediction maps
│   └── predictions/            # CSV prediction results
├── src/                        # Main pipeline code
│   ├── config/                 # Configuration management
│   ├── data_load/              # Data loading utilities
│   ├── data_preprocess/        # Data preprocessing
│   ├── data_validation/        # Data validation
│   ├── features/               # Feature engineering
│   ├── models/                 # Model training and registry
│   ├── pipelines/              # Training and inference pipelines
│   └── utils/                  # Utilities and logging
├── logs/                       # Pipeline logs
├── docs/                       # Documentation
└── tests/                      # Test files
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **Conda or Miniconda** (recommended)
3. **Ollama** (for local LLM capabilities)

### Installation

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd Capstone-Microsoft
```

#### 2. Install Ollama
```bash
# Windows (using winget)
winget install Ollama.Ollama

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

#### 3. Set Up Python Environment

**Option A: Using Conda (Recommended)**
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate Capstone-Microsoft
```

**Option B: Using pip**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
# Test the pipeline
python -m src.main train --environment development
```

## 📊 Data Sources

- **GBIF Occurrence Data**: [Vaccinium membranaceum](https://www.gbif.org/species/9060377)
- **GridMET Climate Data**: [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/gridmet)
- **Elevation Data**: [Open-Elevation API](https://api.open-elevation.com/)
- **Soil Data**: [SoilGrids API](https://www.isric.org/explore/soilgrids)

## 🔧 Usage

### Training Pipeline

Train a new model with your data:

```bash
# Development environment (uses test sample data)
python -m src.main train --environment development

# Production environment (uses full dataset)
python -m src.main train --environment production

# Testing environment
python -m src.main train --environment testing
```

**Training Process:**
1. Loads raw GBIF occurrence data
2. Preprocesses and cleans the data
3. Generates pseudo-absences for balanced training
4. Extracts environmental data (GridMET, elevation, soil)
5. Trains Random Forest or Ensemble model
6. Saves model to registry with versioning
7. Outputs processed and enriched datasets

### Inference Pipeline

Make predictions on new coordinates:

```bash
# Basic inference (uses latest GridMET date)
python -m src.main infer --coordinates 44.5 -116.5 44.6 -116.4 44.7 -116.3 --environment development

# With custom confidence threshold
python -m src.main infer --coordinates 44.5 -116.5 44.6 -116.4 44.7 -116.3 --environment development --confidence-threshold 0.6

# Skip map creation
python -m src.main infer --coordinates 44.5 -116.5 44.6 -116.4 44.7 -116.3 --environment development --no-map
```

**Inference Process:**
1. Validates input coordinates
2. Automatically uses latest available GridMET date
3. Extracts environmental data for coordinates
4. Makes predictions using trained model
5. Saves results to `outputs/predictions/inference_predictions.csv`
6. Creates interactive map in `outputs/maps/prediction_map.html`

### Environment Options

- **`development`**: Uses test sample data, smaller models, debug logging
- **`production`**: Uses full dataset, larger models, info logging
- **`testing`**: Uses test data, minimal models for quick testing
- **`test_sample`**: Uses actual test sample data

## 📁 Outputs

### Training Outputs
- **Processed Data**: `data/processed/huckleberry_processed.csv`
- **Enriched Data**: `data/enriched/huckleberry_final_enriched.csv`
- **Trained Model**: Saved to `models/` with versioning
- **Model Registry**: `models/registry.json`

### Inference Outputs
- **Predictions CSV**: `outputs/predictions/inference_predictions.csv`
- **Interactive Map**: `outputs/maps/prediction_map.html` (if enabled)

## 🔍 Features

### Environmental Variables
- **Climate**: Temperature, precipitation, humidity, wind speed, solar radiation
- **Elevation**: Terrain elevation data
- **Soil**: pH levels from soil surveys
- **Temporal**: Year, month, day, season

### Model Types
- **Random Forest**: Fast, interpretable, good baseline
- **Ensemble**: Combines multiple models for better performance

### Pipeline Features
- **Automatic Data Versioning**: Tracks all data transformations
- **Model Registry**: Versioned model storage and management
- **Flexible Environments**: Different configurations for dev/prod
- **Comprehensive Logging**: Detailed pipeline execution logs
- **Data Validation**: Ensures data quality at each step
- **Error Handling**: Robust error handling and recovery

## 🛠️ Configuration

### Environment Settings

The pipeline uses environment-specific configurations:

```python
# Development (src/config/environments.py)
- Uses test sample data
- Smaller model parameters
- Debug logging
- Specific model file for inference

# Production
- Uses full dataset
- Larger model parameters
- Info logging
- Latest model from registry
```

### Model Settings

```python
# Random Forest
- n_estimators: 100 (dev) / 200 (prod)
- test_size: 0.2
- random_state: 42

# Ensemble
- Combines multiple models
- Automatic hyperparameter optimization
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_basic.py

# Run with coverage
pytest --cov=src tests/
```

## 📈 Performance

### Model Performance
- **Accuracy**: Typically 85-95% on test data
- **Feature Importance**: Environmental variables ranked by importance
- **Cross-validation**: Robust performance evaluation

### Pipeline Performance
- **Training**: 5-15 minutes (depending on data size)
- **Inference**: 30-60 seconds per coordinate set
- **Environmental Extraction**: 2-5 seconds per coordinate

## 🔧 Troubleshooting

### Common Issues

1. **GridMET Connection Issues**
   ```bash
   # Check internet connection
   # Verify Planetary Computer access
   ```

2. **Model Loading Errors**
   ```bash
   # Check model file exists
   # Verify model format compatibility
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size
   # Use smaller test dataset
   ```

### Logs

Check pipeline logs in `logs/` directory:
- `pipeline_dev.log` - Development environment
- `pipeline_prod.log` - Production environment
- `pipeline_test.log` - Testing environment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **GBIF** for occurrence data
- **Microsoft Planetary Computer** for climate data
- **Open-Elevation** for elevation data
- **SoilGrids** for soil data
- **Scikit-learn** for machine learning tools

## 📞 Support

For questions or issues:
1. Check the logs in `logs/` directory
2. Review the documentation in `docs/`
3. Open an issue on GitHub

---

**Note**: This pipeline is designed for research and educational purposes. Always validate predictions with field observations before making management decisions. 