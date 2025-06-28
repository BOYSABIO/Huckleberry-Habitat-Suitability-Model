# Modular Data Processing Pipeline Guide

## Overview

This guide explains the new modular, production-ready data processing pipeline for the Huckleberry Habitat Prediction System. The new structure addresses the messy preprocessing workflow by breaking it down into clean, reusable, and testable components.

## Problem Solved

### Original Messy Workflow
The original preprocessing was scattered across notebooks and scripts with these issues:

1. **Mixed formats**: Code was split between notebooks and scripts
2. **Complex dependencies**: Hard to understand the data flow
3. **Manual steps**: Required manual geocoding and intervention
4. **No error handling**: Failures were hard to debug
5. **No validation**: Data quality issues weren't caught early
6. **Hard to reproduce**: Difficult to rerun the entire pipeline

### New Modular Solution
The new pipeline provides:

1. **Clear separation of concerns**: Each step is a separate module
2. **Reproducible workflow**: Complete pipeline can be run end-to-end
3. **Error handling**: Comprehensive logging and error recovery
4. **Data validation**: Quality checks at each step
5. **Configurable**: Easy to customize for different needs
6. **Testable**: Each component can be tested independently

## Pipeline Architecture

```
Raw GBIF Data → Loader → Preprocessor → Geocoder → Environmental Extractor → Final Dataset
```

### 1. Data Loader (`src/data/loader.py`)
**Purpose**: Load and validate data from various sources

**Key Features**:
- Loads GBIF occurrence data
- Validates data structure and quality
- Provides data summaries and statistics
- Handles different file formats

**Usage**:
```python
from src.data.loader import DataLoader

loader = DataLoader()
raw_data = loader.load_gbif_occurrences("data/raw/occurrence.txt")
summary = loader.get_data_summary(raw_data)
```

### 2. Data Preprocessor (`src/data/preprocessor.py`)
**Purpose**: Clean and filter raw occurrence data

**Key Features**:
- Filters to US geographic bounds
- Removes invalid coordinates and dates
- Handles missing values
- Creates pseudo-absence points
- Removes duplicates

**Usage**:
```python
from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
cleaned_data = preprocessor.clean_occurrence_data(raw_data)
dataset_with_absences = preprocessor.create_pseudo_absences(cleaned_data)
```

### 3. Geocoder (`src/data/geocoder.py`)
**Purpose**: Geocode locations with multiple fallback strategies

**Key Features**:
- Traditional geocoding with Nominatim
- LLM-assisted landmark extraction
- Multiple fallback strategies
- Batch processing with rate limiting
- Comprehensive logging

**Usage**:
```python
from src.data.geocoder import Geocoder

geocoder = Geocoder(use_llm=True)
geocoded_data = geocoder.geocode_dataset(cleaned_data)
```

### 4. Environmental Data Extractor (`src/data/environmental.py`)
**Purpose**: Extract environmental variables from various sources

**Key Features**:
- GridMET climate data extraction
- Elevation data integration
- Soil data integration
- Derived feature creation
- Bounds filtering

**Usage**:
```python
from src.data.environmental import EnvironmentalDataExtractor

extractor = EnvironmentalDataExtractor()
enriched_data = extractor.extract_gridmet_data(geocoded_data)
enriched_data = extractor.add_elevation_data(enriched_data)
enriched_data = extractor.create_environmental_features(enriched_data)
```

### 5. Main Processor (`src/data/processor.py`)
**Purpose**: Orchestrate the entire pipeline

**Key Features**:
- Runs complete pipeline end-to-end
- Configurable processing steps
- Step-by-step processing for debugging
- Comprehensive logging and summaries
- Error recovery and retry logic

**Usage**:
```python
from src.data.processor import DataProcessor

processor = DataProcessor(use_llm=True)
final_data = processor.process_full_pipeline(
    input_file="data/raw/occurrence.txt",
    output_file="HB_enriched.csv"
)
```

## Complete Workflow

### Step 1: Load Raw Data
```python
# Load GBIF occurrence data
raw_data = loader.load_gbif_occurrences()
# Output: DataFrame with raw GBIF columns
```

### Step 2: Clean and Filter
```python
# Clean data and filter to US
cleaned_data = preprocessor.clean_occurrence_data(raw_data)
# Output: Cleaned DataFrame with valid US coordinates
```

### Step 3: Geocode Missing Coordinates
```python
# Geocode locations with missing coordinates
geocoded_data = geocoder.geocode_dataset(cleaned_data)
# Output: DataFrame with complete coordinates
```

### Step 4: Create Pseudo-Absences
```python
# Add pseudo-absence points for modeling
dataset_with_absences = preprocessor.create_pseudo_absences(geocoded_data)
# Output: Balanced dataset with presence/absence data
```

### Step 5: Extract Environmental Data
```python
# Extract GridMET climate data
enriched_data = extractor.extract_gridmet_data(dataset_with_absences)
# Add elevation and soil data
enriched_data = extractor.add_elevation_data(enriched_data)
enriched_data = extractor.add_soil_data(enriched_data)
# Create derived features
enriched_data = extractor.create_environmental_features(enriched_data)
# Output: Final enriched dataset ready for modeling
```

## Configuration Options

### Processor Configuration
```python
processor = DataProcessor(
    data_dir="data",           # Data directory
    use_llm=True              # Use LLM for geocoding
)

# Customize processing
processor.config.update({
    'pseudo_absence_ratio': 3.0,    # Ratio of absences to presences
    'geocoding_batch_size': 100,    # Batch size for geocoding
    'save_intermediate': True       # Save intermediate files
})
```

### Geocoding Configuration
```python
geocoder = Geocoder(
    use_llm=True,             # Enable LLM assistance
    llm_model='llama3:8b'     # LLM model to use
)
```

## Integration with Existing Code

### Replacing Notebook Code
Instead of running cells in notebooks, use the modular pipeline:

**Before (Notebook)**:
```python
# Cell 1: Load data
df = pd.read_csv('occurrence.txt', sep='\t')

# Cell 2: Clean data
df = df[df['countryCode'] == 'US']
df = df.dropna(subset=['decimalLatitude', 'decimalLongitude'])

# Cell 3: Geocode (manual process)
# ... manual geocoding with LLM ...

# Cell 4: Extract environmental data
# ... GridMET extraction ...
```

**After (Modular)**:
```python
from src.data.processor import DataProcessor

processor = DataProcessor()
final_data = processor.process_full_pipeline()
```

### Integrating Existing Scripts
You can integrate your existing elevation and soil scripts:

```python
# In environmental.py, replace placeholder methods:
def add_elevation_data(self, df):
    # Import your existing elevation.py logic
    from src.elevation import extract_elevation_data
    return extract_elevation_data(df)

def add_soil_data(self, df):
    # Import your existing soil_data.py logic
    from src.soil_data import extract_soil_data
    return extract_soil_data(df)
```

## Error Handling and Debugging

### Step-by-Step Processing
For debugging, run individual steps:

```python
# Run specific steps only
results = processor.process_step_by_step(
    steps=['load', 'clean', 'geocode']
)

# Inspect intermediate results
print(f"Cleaned data: {len(results['cleaned'])} records")
print(f"Geocoded data: {len(results['geocoded'])} records")
```

### Logging and Monitoring
Each module provides comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Run pipeline with detailed logging
processor.process_full_pipeline()
```

### Data Validation
Validate data at each step:

```python
# Validate final dataset
is_valid = loader.validate_data(
    final_data, 
    expected_columns=['decimallatitude', 'decimallongitude', 'occurrence']
)
```

## Performance Optimization

### Batch Processing
Process data in batches to manage memory:

```python
# Configure batch sizes
processor.config['geocoding_batch_size'] = 50
```

### Parallel Processing
For large datasets, consider parallel processing:

```python
# Process in parallel (future enhancement)
processor.config['parallel_processing'] = True
processor.config['n_jobs'] = 4
```

### Caching
Save intermediate results to avoid reprocessing:

```python
# Save intermediate files
processor.config['save_intermediate'] = True
```

## Testing and Validation

### Unit Tests
Test individual components:

```python
# Test preprocessor
def test_preprocessor():
    preprocessor = DataPreprocessor()
    sample_data = create_sample_data()
    cleaned = preprocessor.clean_occurrence_data(sample_data)
    assert len(cleaned) > 0
```

### Integration Tests
Test the complete pipeline:

```python
# Test complete pipeline
def test_full_pipeline():
    processor = DataProcessor()
    result = processor.process_full_pipeline(skip_environmental=True)
    assert 'occurrence' in result.columns
```

## Migration Guide

### From Notebooks to Scripts
1. **Identify notebook cells**: Map each cell to a module
2. **Extract functions**: Move logic to appropriate modules
3. **Update imports**: Replace notebook imports with module imports
4. **Test incrementally**: Test each module individually

### From Scripts to Modules
1. **Refactor existing scripts**: Break into smaller functions
2. **Add error handling**: Wrap in try-catch blocks
3. **Add logging**: Replace print statements with logging
4. **Add validation**: Validate inputs and outputs

## Best Practices

### Code Organization
- Keep modules focused on single responsibilities
- Use clear, descriptive function names
- Add comprehensive docstrings
- Follow PEP 8 style guidelines

### Error Handling
- Catch specific exceptions, not generic ones
- Provide meaningful error messages
- Log errors with context
- Implement retry logic for transient failures

### Data Validation
- Validate inputs at each step
- Check data types and ranges
- Handle missing values appropriately
- Document data quality issues

### Configuration
- Use configuration files for settings
- Provide sensible defaults
- Allow runtime configuration
- Document all configuration options

## Future Enhancements

### Planned Improvements
1. **Parallel processing**: Process large datasets faster
2. **Caching**: Cache expensive operations
3. **Monitoring**: Add performance monitoring
4. **API integration**: REST API for pipeline execution
5. **Web interface**: GUI for pipeline management

### Extensibility
The modular design makes it easy to add new features:

```python
# Add new environmental data source
class NewEnvironmentalExtractor:
    def extract_new_data(self, df):
        # Implementation
        pass

# Integrate into pipeline
processor.environmental_extractor.new_extractor = NewEnvironmentalExtractor()
```

## Conclusion

The new modular pipeline provides a clean, maintainable, and extensible solution to the messy preprocessing workflow. It addresses all the original issues while providing a solid foundation for future development.

Key benefits:
- **Maintainable**: Clear separation of concerns
- **Testable**: Each component can be tested independently
- **Reproducible**: Complete pipeline can be run end-to-end
- **Configurable**: Easy to customize for different needs
- **Extensible**: Easy to add new features and data sources

This modular approach transforms the complex, manual preprocessing workflow into a clean, automated pipeline that's ready for production use. 