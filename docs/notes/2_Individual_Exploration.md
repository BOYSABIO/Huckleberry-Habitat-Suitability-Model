# Capstone Strategy: Weather Prediction & City Planning MVPs

This document outlines a phased, scalable capstone strategy based on two high-potential tracks: weather-insurance intelligence and smart city planning, beginning with technically focused MVPs.

---

## Track 1: Crop & Property Insurance – Phase 1: Weather Prediction Engine

### Why This Is the Right Start
- Downstream apps depend on accurate, localized forecasts.
- Builds core technical credibility.
- Fast-to-demo and explain.

### MVP Scope
- **Goal**: Predict daily/weekly weather (temp, precipitation, humidity, wind) at selected coordinates.
- **Data**: Microsoft gridMET historical climate dataset.
- **Model Options**:
  - Baseline: ARIMA, SARIMAX.
  - Intermediate: XGBoost, Random Forests.
  - Advanced: LSTM/GRU, Temporal Fusion Transformer.
- **Deliverable**:
  - Input: lat/lon selector
  - Output:
    - Multi-day forecasts
    - Confidence intervals
    - Historical visualizations

### Expansion Path
- Predict crop loss using yield + forecast data.
- Build a regional insurance risk index.
- Visual heatmaps + underwriting support dashboard.

---

## Track 2: Future City Planning – Phase 1: Data Center Placement Recommender

### Why This Is a Good Starting Point
- High relevance (data centers = energy-intensive).
- Easier to scope than simulating entire city dynamics.
- Great business/industry pitch.

### MVP Scope
- **Goal**: Score US regions by their climate suitability for data center placement.
- **Criteria**:
  - Temp stability
  - Fire/flood risk
  - Solar potential
- **Data**: gridMET + optional overlays (land price, power infra).
- **Model Options**:
  - Weighted scoring system
  - Clustering to find “ideal climate” zones
- **Deliverable**:
  - Interactive map of recommendations
  - User-defined weighting sliders
  - Exportable report

### Expansion Path
- Add zoning, population forecasts
- Expand into broader CitySim-X
- LLM-based query interface for planning insights

---

## Technical Stack

- **Backend/Data**: xarray, pandas, numpy, PostgreSQL/PostGIS (if needed)
- **Modeling**: statsmodels, sklearn, PyTorch, XGBoost
- **Frontend/UI**: Streamlit, Dash, or React + Leaflet
- **Mapping**: GeoPandas, Folium, Deck.gl, CesiumJS

---

## Phase Table

| Phase         | Objective                                  | Track 1 (Insurance)         | Track 2 (City Planning)      |
|---------------|---------------------------------------------|-----------------------------|------------------------------|
| **Now**       | MVP + Core Modeling                         | Weather Forecasting         | Data Center Recommender      |
| **Mid-stage** | Thematic Expansion                          | Crop Loss / Home Risk       | Build Zone Suggestion Tool   |
| **Endgame**   | Visionary Product & UX                      | Insurance SaaS Platform     | CitySim-X Digital Twin       |

---

This phased strategy keeps the vision big while anchoring the execution in technically feasible milestones.
