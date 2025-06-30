# Huckleberry Habitat Prediction - Inference Guide

## Overview

This guide explains the inference approach for the huckleberry habitat prediction pipeline. The inference system is designed to predict habitat suitability for geographic areas of interest, rather than just processing new occurrence data.

## Key Concepts

### 1. Training vs. Inference Pipeline

**Training Pipeline** (for model development):
- Processes raw GBIF occurrence data
- Cleans, geocodes, and validates data
- Generates pseudo-absences
- Extracts environmental features
- Trains the model

**Inference Pipeline** (for predictions):
- Takes geographic coordinates of interest
- Extracts environmental data for those coordinates
- Applies feature engineering
- Makes habitat suitability predictions

### 2. Feature Engineering Consistency

Both pipelines use the same feature engineering steps to ensure consistency:
- **Environmental data extraction**: GridMET climate, elevation, soil pH
- **Season feature**: `season_num` (1=Winter, 2=Spring, 3=Summer, 4=Fall)
- **Feature selection**: Excludes `gbifID`, `decimalLatitude`, `decimalLongitude` during training

## Usage Examples

### Basic Habitat Suitability Prediction

```python
from src.inference.predict import predict_habitat_suitability
from datetime import datetime

# Define geographic area of interest
lat_min, lat_max = 44.5, 45.0  # Northern Idaho
lon_min, lon_max = -116.5, -116.0

# Set target date (summer when huckleberries are typically found)
target_date = datetime(2020, 7, 15)

# Run prediction
results = predict_habitat_suitability(
    model_path="models/huckleberry_model.joblib",
    lat_min=lat_min,
    lat_max=lat_max,
    lon_min=lon_min,
    lon_max=lon_max,
    resolution=0.01,  # 1km resolution
    target_date=target_date,
    output_path="outputs/habitat_prediction.csv"
)
```

### Creating Custom Prediction Grids

```python
from src.inference.predict import create_prediction_grid, prepare_inference_data

# Create a custom grid
grid_df = create_prediction_grid(
    lat_min=44.0, lat_max=45.0,
    lon_min=-117.0, lon_max=-116.0,
    resolution=0.005  # 500m resolution
)

# Prepare data for inference
inference_df = prepare_inference_data(grid_df, target_date=datetime(2020, 7, 15))
```

## Best Practices

### 1. Geographic Bounds
- Ensure your area of interest is within GridMET bounds (continental US)
- Consider computational resources when choosing resolution
- Start with smaller areas for testing

### 2. Temporal Considerations
- Choose dates when huckleberries are typically present (summer months)
- Consider seasonal variations in habitat suitability
- Use consistent dates for comparative analyses

### 3. Resolution Selection
- **0.01 degrees** ≈ 1km (good for regional analysis)
- **0.005 degrees** ≈ 500m (higher detail, more computational cost)
- **0.02 degrees** ≈ 2km (faster, less detail)

### 4. Data Quality
- The inference pipeline automatically filters coordinates outside GridMET bounds
- Environmental data extraction may fail for some coordinates (handled gracefully)
- Check prediction confidence scores for reliability

## Output Format

The prediction results include:

```python
{
    'decimalLatitude': float,      # Grid point latitude
    'decimalLongitude': float,     # Grid point longitude
    'prediction': int,             # 0=unsuitable, 1=suitable
    'probability_absence': float,  # Probability of absence
    'probability_presence': float, # Probability of presence
    'elevation': float,            # Elevation in meters
    'soil_ph': float,              # Soil pH
    'air_temperature': float,      # Air temperature (°C)
    'precipitation_amount': float, # Precipitation (mm)
    'season_num': int              # Season (1-4)
}
```

## Error Handling

### Common Issues and Solutions

1. **Model file not found**
   - Train a model first using the main pipeline
   - Check the model path is correct

2. **No coordinates within GridMET bounds**
   - Ensure your area is within continental US
   - Check coordinate system (should be WGS84)

3. **Environmental data extraction failures**
   - Check internet connection
   - Verify GridMET service availability
   - Consider using cached environmental data

4. **Memory issues with large areas**
   - Reduce resolution
   - Process area in smaller chunks
   - Use more efficient data structures

## Advanced Usage

### Batch Processing Multiple Areas

```python
import pandas as pd
from src.inference.predict import predict_habitat_suitability

# Define multiple areas
areas = [
    {'name': 'Northern Idaho', 'bounds': (44.5, 45.0, -116.5, -116.0)},
    {'name': 'Central Idaho', 'bounds': (44.0, 44.5, -115.5, -115.0)},
    {'name': 'Southern Idaho', 'bounds': (43.5, 44.0, -114.5, -114.0)}
]

# Process each area
for area in areas:
    lat_min, lat_max, lon_min, lon_max = area['bounds']
    results = predict_habitat_suitability(
        model_path="models/huckleberry_model.joblib",
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        output_path=f"outputs/{area['name']}_prediction.csv"
    )
```

### Seasonal Analysis

```python
# Analyze habitat suitability across seasons
seasons = [
    datetime(2020, 1, 15),  # Winter
    datetime(2020, 4, 15),  # Spring
    datetime(2020, 7, 15),  # Summer
    datetime(2020, 10, 15)  # Fall
]

for season in seasons:
    results = predict_habitat_suitability(
        model_path="models/huckleberry_model.joblib",
        lat_min=44.5, lat_max=45.0,
        lon_min=-116.5, lon_max=-116.0,
        target_date=season,
        output_path=f"outputs/seasonal_{season.strftime('%m')}_prediction.csv"
    )
```

## Integration with GIS

The output CSV files can be easily imported into GIS software:

1. **QGIS**: Use "Add Delimited Text Layer"
2. **ArcGIS**: Use "XY Table to Point" tool
3. **R**: Use `read.csv()` and `spatial` packages
4. **Python**: Use `geopandas` for spatial analysis

## Performance Considerations

- **Grid size**: Larger areas require more computational resources
- **Resolution**: Higher resolution = more points = longer processing time
- **Environmental data**: GridMET extraction is the most time-consuming step
- **Caching**: Consider caching environmental data for repeated analyses

## Troubleshooting

### Debug Mode
Enable detailed logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validation
Always validate your results:
- Check prediction distributions
- Verify environmental data ranges
- Compare with known occurrence data
- Use cross-validation during model training

## Conclusion

This inference approach provides a flexible and powerful way to predict huckleberry habitat suitability across geographic areas. The key is maintaining consistency between training and inference feature engineering while allowing for geographic and temporal flexibility in predictions. 