# Capstone Project Ideas – Data Science Master's

This document collects a set of project ideas discussed as part of evaluating potential directions for a final capstone project with Microsoft Planetary Computer data. Focus areas include agriculture, real estate, logistics, and climate applications. Projects span ML, CV, RL, geolocation, and LLMs.

---

## 🌾 Agriculture & Climate Projects

### 1. **Crop Yield Forecasting with Satellite + Weather Data**
- **Description**: Predict crop yield using NDVI or other vegetation indices from satellite imagery, paired with historical weather data.
- **Tech**: CNNs for image analysis, time series forecasting, data fusion.
- **Deliverable**: Forecast dashboard with geospatial input.

### 2. **Smart Irrigation System (Reinforcement Learning)**
- **Description**: Use RL to optimize irrigation scheduling based on crop type, weather forecast, and soil moisture.
- **Tech**: Custom OpenAI Gym environment, RL (e.g., PPO or DQN).
- **Reward**: Maximize yield, minimize water usage.

### 3. **Adaptive Pest Management Agent (RL + CV)**
- **Description**: Detect pests via drone or field images, and use RL to decide treatment timing.
- **Tech**: CV for pest detection, RL for intervention decisions.

### 4. **Real-Time Agricultural Alert System**
- **Description**: Mobile-first app that pushes alerts for extreme weather events relevant to crop safety.
- **Tech**: Forecast models + geolocation + push notification system.

### 5. **LLM-Based Agricultural Assistant**
- **Description**: Natural language tool for farmers to ask questions like "What should I plant this season?"
- **Tech**: Retrieval-Augmented Generation (RAG) using climate + crop data, small LLMs or OpenAI API.

---

## 🏘️ Real Estate & Risk

### 6. **Climate Risk Score for Real Estate or Agriculture**
- **Description**: Build a composite risk score for properties based on weather patterns, floods, droughts.
- **Tech**: ML models, geospatial data processing, map-based visualization.

### 7. **Property Resilience Index**
- **Description**: Develop a scoring system to evaluate property climate resilience (e.g., elevation, urban heat island, historical exposure).
- **Tech**: Composite scoring, clustering (k-means, t-SNE), web dashboard.

---

## 🚚 Logistics & Operations

### 8. **Sustainable Logistics Planning**
- **Description**: Optimize delivery or farm routes based on emissions, weather, and road conditions.
- **Tech**: Optimization + predictive modeling.

### 9. **Climate-Aware Route Planner (RL)**
- **Description**: RL agent selects delivery or equipment routes dynamically, avoiding high-risk zones.
- **Tech**: RL with geospatial environment; reward based on timeliness + carbon cost.

### 10. **Smart Tractor Assistant**
- **Description**: Tool to optimize tractor movement based on soil and climate data.
- **Tech**: Path optimization, map overlays, climate-aware constraints.

---

## 🧠 Technical Notes

- **Feasibility**: All projects scoped to fit a PC with RTX 3060 + 64GB RAM.
- **CV Ready**: Image processing manageable if batched/downsampled.
- **RL Practicality**: RL is realistic with simplified environments and limited action/state spaces.
- **LLMs**: Use API-based models or quantized local models for NLP tasks.
- **Data**: Use Microsoft Planetary Computer (satellite, raster, climate data), possibly enriched with phone GPS, API weather feeds, or public ag datasets.

---

## ✅ Next Steps

- Pick 1–2 favorite ideas.
- Draft a mini-proposal (goal, data, tech stack, impact).
- Start with exploratory data analysis or model prototyping.
- Scope MVP and build incrementally.

---

Let this list evolve as you experiment!
