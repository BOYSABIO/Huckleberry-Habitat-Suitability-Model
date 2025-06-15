
# 🍇 Huckleberry Habitat & Yield Prediction Project Plan

A step-by-step breakdown of the full pipeline from raw data to model deployment.

---

## STAGE 1: DATA CLEANING & PREPARATION

### 1. Clean GBIF Occurrence Data (`occurrence.txt`)
- Load with `pandas.read_csv(..., sep='\t')`
- Drop rows with missing or 0-coordinates
- Keep rows with `basisOfRecord` in: `"HumanObservation"`, `"PreservedSpecimen"`
- Convert `eventDate` to datetime or fallback to `year`
- Filter to recent years: 2000–present (to match GridMET)

### 2. Create a GeoDataFrame
- Use `geopandas` to convert lat/lon into geometry
- Plot a sample of points to ensure they fall in North America

### 3. (Optional) Generate Pseudo-Absences
- Randomly sample points within the same general lat/lon bounding box
- Ensure they are far enough from any known positive point (e.g., >10km)
- Label them as `"absent"` for classification

---

## STAGE 2: ENVIRONMENTAL FEATURE ENGINEERING

### 4. Extract Static Features (per location)
- Soil pH (from SoilGrids or SSURGO)
- Elevation (from SRTM or USGS)
- Slope or aspect (optional)
- Canopy density or NDVI (from MODIS)
- Land cover class (optional)

### 5. Extract Dynamic Features (per location and year)
- Use GridMET:
  - Annual average temperature
  - Total precipitation
  - Solar radiation
  - Vapor pressure deficit
- Build these per year or as rolling averages (e.g., past 3-year window)

### 6. Create Final Modeling Dataset
- One row per location (+ optional year)
- Include:
  - Lat, Lon
  - Presence/absence label (or synthetic yield)
  - Environmental features

---

## STAGE 3: SYNTHETIC YIELD GENERATION (Optional)
- Create a scoring function using:
  - Solar radiation
  - Precipitation
  - Soil pH
  - Canopy density
- Map scores to yield buckets (e.g., 0–200 lbs/acre)
- Add noise or uncertainty to simulate real-world variation

---

## STAGE 4: MODEL EXPERIMENTATION

### 7. Baseline ML Models (Tabular)
- Random Forest
- XGBoost
- Logistic Regression
- LightGBM (bonus)
> Evaluate using Accuracy / F1 (classification) or MAE / R² (regression)

### 8. Neural Network (Fully Connected / Feedforward)
- Normalize input features
- Use simple MLP (2–3 layers, ReLU, BatchNorm, Dropout)
- Output: Sigmoid (binary) or Linear (regression)

### 9. LSTM (Advanced, only if temporal)
Only use if time-sequences are engineered (e.g., 3-year trends)

### 10. GNN (Optional Stretch Goal)
Use for spatial similarity or habitat expansion prediction

---

## STAGE 5: EVALUATION & VISUALIZATION

- Confusion matrix (if classification)
- Feature importance plot
- Spatial heatmap of predicted suitability or yield
- Overlay predictions with known harvest zones

---

## STAGE 6: MVP APPLICATION (Optional)
- Streamlit web app
- Region selector → predicted habitat/yield
- Climate layer toggles & model confidence

---

## Summary on Neural Networks

| Neural Net Type | Use it if... |
|------------------|-----------------------------|
| MLP              | You have ~1k+ labeled tabular samples |
| LSTM             | You’ve created time-series per location |
| GNN              | You want to model spatial dependencies |

---
